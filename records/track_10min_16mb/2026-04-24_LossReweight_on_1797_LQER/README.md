# Loss Reweighting by Token Predictability — on PR #1797 LQER Asym base

## Summary

Original-research experiment: applies **predictability-weighted cross-entropy** during main training on top of PR #1797 (dexhunter, 2026-04-24), the current NN-only legal frontier at **1.06157 3-seed mean**. Hypothesis: under a fixed 600s training budget, biasing gradient toward hard tokens lets the model spend capacity where the BPB metric actually accumulates loss, instead of over-fitting easy bigrams.

Not a PR mash. The reweight loss is a new mechanism not present in any open PR (verified 2026-04-24; PR #1787's "Fused CE" is a Triton speed kernel, NOT per-token loss reweighting — their kernel already exposes `reduction="none"`, which this submission reuses).

## Base

- **PR #1797** (dexhunter, 2026-04-24): PR #1787 native base + Smear Gate + LQER Asymmetric rank-4 post-GPTQ correction
- PR #1787 base provides: CaseOps + SparseAttnGate + PolarNS + MIN_LR + FusedCE (Triton) + PR #1767 TTT warm-start-A
- 3-seed mean: **val_bpb 1.06157** (std 0.00066)
- Single change: `forward()` loss now supports a per-token weighted CE controlled by `LOSS_REWEIGHT_ALPHA`

## Method

Per-token NLL is rescaled by a detached, clipped, batch-normalized weight:

```python
if alpha == 0.0:
    # identical to PR #1797 (fused path if FUSED_CE_ENABLED=1, else eager)
    return softcapped_cross_entropy(logits, targets, softcap, reduction="mean")
nll = softcapped_cross_entropy(logits, targets, softcap, reduction="none")  # per-row
ramp = clamp(step / warmup_steps, max=1.0)
a_eff = alpha * ramp                            # tensor scalar, gradient-free
ratio = (nll.detach() / nll.detach().mean()).clamp(clip_lo, clip_hi)
w = ratio.pow(a_eff)
w = w / w.mean()                                # mean-1 normalize → loss scale stable
loss = (w * nll).mean()
```

Properties:
- `alpha == 0.0` is a constant Python branch → cheap fall-through to the PR #1797 fused CE path (or eager if `FUSED_CE_ENABLED=0`). Byte-for-byte identical numerics when disabled.
- `alpha > 0`: upweight tokens with above-average NLL ("hard tokens"). Standard focal-style intuition adapted to LM training.
- `alpha < 0`: upweight easy tokens (anti-focal). Worth sweeping as a control.
- Weights are detached so no second-order gradient through the weighting function.
- Mean-1 normalization keeps the effective loss magnitude comparable across alpha values, so the optimizer's LR doesn't need retuning per alpha.
- Warmup ramp (default 500 steps) avoids early-training collapse onto noisy NLL spikes before the model has a baseline.
- Clip range `[0.1, 10.0]` prevents single tokens from dominating the batch.

### FusedCE compatibility

PR #1787/#1797's Triton FusedCE already exposes `reduction="none"`. The reweight path calls `softcapped_cross_entropy(..., reduction="none")`, multiplies by detached per-row weights, and means — all in eager. The fused matmul/softmax/LSE pass is preserved; only the scalar reduction is replaced. Backward: autograd handles the `(w * nll).mean()` chain correctly because the kernel's registered backward operates on arbitrary per-row `grad_losses`.

Implementation points:
- Hyperparameters block (~line 368): adds 4 env vars.
- `GPT.__init__` (~line 1152): stores hparams and registers `_train_step` buffer.
- `GPT.forward()` (~line 1413): branches on `alpha == 0.0`.
- `step_fn` (~line 3206): fills `base_model._train_step` each step.

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

- Base PR #1797 3-seed mean: **1.06157**
- Beats base (validates mechanism): **≤ 1.06157 − seed_std(0.00066) = ≤ 1.06091**
- New legal record (≥0.005 nat improvement at p<0.01): **≤ 1.05657**

## Reproduction

One-time setup: same as PR #1797 (CaseOps shards + byte sidecar via `prepare_caseops_data.py`, FA3 wheel, etc.).

Single seed at `alpha=0.0` (sanity, must reproduce PR #1797):
```bash
NCCL_NET=Socket DATA_DIR=./data CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 MATRIX_LR=0.026 MIN_LR=0.1 \
  FUSED_CE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 TTT_WARM_START_A=1 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  LOSS_REWEIGHT_ALPHA=0.0 \
  SEED=42 RUN_ID=train_seed42_a0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > train_seed42_a0.log 2>&1
```

Single seed at `alpha=1.0`:
```bash
# same env as above, plus:
LOSS_REWEIGHT_ALPHA=1.0 SEED=42 RUN_ID=train_seed42_a1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > train_seed42_a1.log 2>&1
```

Sweep plan (single seed first, 3-seed if winning alpha found):
```bash
for a in 0.0 0.5 1.0 2.0 -0.5; do
  LOSS_REWEIGHT_ALPHA=$a SEED=42 RUN_ID=train_seed42_a${a} \
    # ... full env as above ...
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > train_seed42_a${a}.log 2>&1
done
```

## Decision rules

- **alpha=0 control (sanity)** must reproduce PR #1797 within seed noise (≤ 1.06157 ± 0.001). Confirms the patch is a no-op when disabled.
- **best alpha** must beat 1.06157 by > seed std (~0.00066) on a single seed → trigger 3-seed validation on `{0, 42, 1337}`.
- If no alpha beats base at single seed → mechanism is null on this stack. Document as throwaway.

## Results

_Pending — alpha=0 sanity run first, then sweep._
