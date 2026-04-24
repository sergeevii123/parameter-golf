# Loss Reweighting by Token Predictability — on PR #1775 NoGates base

## Summary

Original-research mechanism test: applies **predictability-weighted cross-entropy** during main training on top of PR #1775 (dentity007, 2026-04-22), the plain SP8192 + Multi-Phase Global SGD TTT base. Hypothesis: under a fixed 600s training budget, biasing gradient toward hard tokens lets the model spend capacity where the BPB metric actually accumulates loss, instead of over-fitting easy bigrams.

**Why this base, not PR #1797:** PR #1797 (LQER, current NN frontier 1.06157) needs CaseOps shards (~19 GB extra tokenized data). On a 50 GB workstation that does not fit alongside the existing plain SP8192 shards. PR #1775 reuses the plain SP8192 shards already on disk. Mechanism validation here is cheap; if a winning alpha is found, port to PR #1797 / CaseOps stack on a larger box.

Not a PR mash. The reweight loss is a new mechanism not present in any open PR (verified 2026-04-24; PR #1787's "Fused CE" is a Triton speed kernel that exposes `reduction="none"` but does NOT do per-token loss reweighting).

## Base

- **PR #1775** (dentity007, 2026-04-22): SP8192 + No Gates + Multi-Phase Global SGD TTT
- 3-seed mean: **val_bpb 1.07285**
- Single change: `forward()` loss now supports a per-token weighted CE controlled by `LOSS_REWEIGHT_ALPHA`

## Method

Per-token NLL is rescaled by a detached, clipped, batch-normalized weight:

```python
if alpha == 0.0:
    return F.cross_entropy(flat_logits, flat_targets, reduction="mean")  # bit-identical to PR #1775
nll = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # per-row
ramp = clamp(step / warmup_steps, max=1.0)
a_eff = alpha * ramp                            # tensor scalar, gradient-free
ratio = (nll.detach() / nll.detach().mean()).clamp(clip_lo, clip_hi)
w = ratio.pow(a_eff)
w = w / w.mean()                                # mean-1 normalize → loss scale stable
loss = (w * nll).mean()
```

Properties:
- `alpha == 0.0` is a constant Python branch → returns the original `F.cross_entropy(reduction="mean")`. Byte-for-byte identical to PR #1775 when disabled.
- `alpha > 0`: upweight tokens with above-average NLL ("hard tokens"). Standard focal-style intuition adapted to LM training.
- `alpha < 0`: upweight easy tokens (anti-focal). Worth sweeping as a control.
- Weights are detached so no second-order gradient through the weighting function.
- Mean-1 normalization keeps the effective loss magnitude comparable across alpha values, so the optimizer's LR doesn't need retuning per alpha.
- Warmup ramp (default 500 steps) avoids early-training collapse onto noisy NLL spikes before the model has a baseline.
- Clip range `[0.1, 10.0]` prevents single tokens from dominating the batch.

Implementation points (line numbers in this file):
- Hyperparameters block (line 191): adds 4 env vars.
- `GPT.__init__` (line 877): stores hparams and registers `_train_step` buffer.
- `GPT.forward` (line 1128): branches on `alpha == 0.0`.
- `step_fn` (line 3061): fills `base_model._train_step` each step.

## Why this could help BPB specifically

- BPB metric aggregates per-token NLL uniformly across the val stream.
- Standard CE training already minimizes mean NLL — but with a finite training budget, the optimizer's gradient is dominated by *whichever tokens currently have the largest loss times their frequency*. This typically converges fast on common easy tokens and stalls on the long tail.
- Reweighting shifts gradient toward tokens that are still losing — likely the long tail that contributes most to the residual BPB gap.
- Risk: high-NLL tokens may be *inherently* unpredictable (random numbers, proper nouns, non-English), in which case extra gradient is wasted.
- The clip range and warmup are designed to bound the worst-case loss but won't fully eliminate it. Sweeping alpha tells us empirically.

## Hyperparameters

| Env var | Default | Range to sweep |
|---|---|---|
| `LOSS_REWEIGHT_ALPHA` | `0.0` | `{0.0, 0.5, 1.0, 2.0, -0.5}` |
| `LOSS_REWEIGHT_CLIP_LO` | `0.1` | fixed for sweep |
| `LOSS_REWEIGHT_CLIP_HI` | `10.0` | fixed for sweep |
| `LOSS_REWEIGHT_WARMUP_STEPS` | `500` | fixed for sweep |

## Targets

- Base PR #1775 3-seed mean: **1.07285**
- Beats base (validates mechanism): **< 1.07285** by > seed std
- New legal record (must beat current 1.06157 frontier by ≥0.005 nat at p<0.01): **≤ 1.05657**

This base is not record-eligible by itself (1.07285 > 1.06157). The goal here is **mechanism validation** — if alpha>0 reliably improves on PR #1775 base, port the patch to PR #1797 / CaseOps on a larger box and re-test for record eligibility.

## Requirements

**Hardware**: 8× H100 SXM (FA3 is Hopper-only; Ampere/Ada GPUs will fail on `from flash_attn_interface import flash_attn_func`).

**Python**: ≥ 3.12.

**Packages**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece triton numpy brotli
pip install flash_attn_3 --no-deps \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

**Data**: plain SP8192 FineWeb shards in `./data/datasets/fineweb10B_sp8192/` (~19 GB). No CaseOps sidecar needed — this base does not use CaseOps.

## Reproduction

All commands assume **CWD = repo root** (so `./data/datasets/fineweb10B_sp8192/...` and `./data/tokenizers/fineweb_8192_bpe.model` resolve correctly). Pass the script by full path.

Single seed at `alpha=0.0` (sanity, must reproduce PR #1775):
```bash
SCRIPT=records/track_10min_16mb/2026-04-24_LossReweight_on_1775_NoGates/train_gpt.py
NCCL_NET=Socket DATA_DIR=./data \
  LOSS_REWEIGHT_ALPHA=0.0 \
  SEED=42 RUN_ID=train_seed42_a0 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_a0.log 2>&1
```

Single seed at `alpha=1.0`:
```bash
SCRIPT=records/track_10min_16mb/2026-04-24_LossReweight_on_1775_NoGates/train_gpt.py
NCCL_NET=Socket DATA_DIR=./data \
  LOSS_REWEIGHT_ALPHA=1.0 \
  SEED=42 RUN_ID=train_seed42_a1 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_a1.log 2>&1
```

Sweep plan (single seed first, 3-seed if winning alpha found):
```bash
SCRIPT=records/track_10min_16mb/2026-04-24_LossReweight_on_1775_NoGates/train_gpt.py
for a in 0.0 0.5 1.0 2.0 -0.5; do
  NCCL_NET=Socket DATA_DIR=./data \
    LOSS_REWEIGHT_ALPHA=$a SEED=42 RUN_ID=train_seed42_a${a} \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    > train_seed42_a${a}.log 2>&1
done
```

## Decision rules

- **alpha=0 control (sanity)** must reproduce PR #1775 within seed noise. Confirms the patch is a no-op when disabled.
- **best alpha** must beat 1.07285 by > seed std on a single seed → trigger 3-seed validation on `{0, 42, 1337}`, then port to PR #1797 base.
- If no alpha beats base at single seed → mechanism is null. Document as throwaway.

## Results

_Pending — alpha=0 sanity run first, then sweep._
