# Loss Reweighting by Token Predictability — on PR #1769 CaseOps base

## Summary

Original-research experiment: applies **predictability-weighted cross-entropy** during main training on top of PR #1769's CaseOps stack (current legal-frontier base, 1.06453 5-seed mean). Hypothesis: under a fixed 600s training budget, biasing gradient toward hard tokens lets the model spend capacity where the BPB metric actually accumulates loss, instead of over-fitting easy bigrams.

Not a PR mash. The reweight loss is a new mechanism not present in any open PR (verified 2026-04-22).

## Base

- **PR #1769** (dexhunter, 2026-04-22): SP8192 + CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT + MLPClip12
- 5-seed mean: **val_bpb 1.06453** (std 0.00068)
- Single change: `forward()` loss now supports a per-token weighted CE controlled by `LOSS_REWEIGHT_ALPHA`

## Method

Per-token NLL is rescaled by a detached, clipped, batch-normalized weight:

```python
nll = F.cross_entropy(logits, target, reduction="none")
ramp = clamp(step / warmup_steps, max=1.0)
a_eff = alpha * ramp                          # tensor scalar, gradient-free
ratio = (nll.detach() / nll.detach().mean()).clamp(clip_lo, clip_hi)
w = ratio.pow(a_eff)
w = w / w.mean()                              # mean-1 normalize → loss scale stable
loss = (w * nll).mean()
```

Properties:
- `alpha == 0.0` is the default and is a constant Python branch → cheap fall-through to standard `F.cross_entropy(reduction="mean")`. No perf cost when disabled.
- `alpha > 0`: upweight tokens with above-average NLL ("hard tokens"). Standard focal-style intuition adapted to LM training.
- `alpha < 0`: upweight easy tokens (anti-focal). Worth sweeping as a control.
- Weights are detached so no second-order gradient through the weighting function.
- Mean-1 normalization keeps the effective loss magnitude comparable across alpha values, so the optimizer's LR doesn't need retuning per alpha.
- Warmup ramp (default 500 steps) avoids early-training collapse onto noisy NLL spikes before the model has a baseline.
- Clip range `[0.1, 10.0]` prevents single tokens from dominating the batch.

Implementation: `train_gpt.py` line ~1132 (`forward`), with `_train_step` buffer mutated in `step_fn` (line ~2811).

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

- Base PR #1769 5-seed mean: **1.06453**
- Beats base (validates mechanism): **≤ 1.06453**
- New legal record (≥0.005 nat improvement at p<0.01): **≤ 1.05953**

## Reproduction

One-time setup: same as PR #1769 (CaseOps shards + byte sidecar via `prepare_caseops_data.py`, FA3 wheel, etc.).

Single seed at `alpha=1.0`:
```bash
SEED=42 RUN_ID=train_seed42_a1 \
  LOSS_REWEIGHT_ALPHA=1.0 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-22_LossReweight_Predictability_CaseOps/train_gpt.py
```

Sweep plan (single seed first, 3-seed if winning alpha found):
```bash
for a in 0.0 0.5 1.0 2.0 -0.5; do
  LOSS_REWEIGHT_ALPHA=$a SEED=42 RUN_ID=train_seed42_a${a} ... torchrun ...
done
```

## Decision rules

- **alpha=0 control (sanity)** must reproduce PR #1769 within seed noise (≤ 1.0653 ± 0.001). Confirms the patch is a no-op when disabled.
- **best alpha** must beat 1.06453 by > seed std (~0.0007) on a single seed → trigger 3-seed validation.
- If no alpha beats base at single seed → mechanism is null on this stack. Document as throwaway.

## Results

_Pending — alpha=0 sanity run first, then sweep._
