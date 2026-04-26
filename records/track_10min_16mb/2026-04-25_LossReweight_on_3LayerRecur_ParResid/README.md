# Loss Reweighting by Token Predictability — on PR #1413 (3-Layer Recurrence + Parallel Residuals)

## Summary

Mechanism replication study: applies **predictability-weighted cross-entropy** during main training on top of PR #1413 (`bigbag`/`dexhunter`, 2026-04-09): SP8192 + 3-Layer Depth Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal Score-First TTT (3-seed mean **val_bpb 1.0810**, currently merged to `main`).

Hypothesis: under a fixed 600s training budget, biasing per-token gradient toward hard tokens shifts capacity to long-tail residuals where BPB still accumulates loss, instead of over-fitting easy bigrams.

## Prior art (acknowledged)

This mechanism family has been tried twice before with **negative results**:

- **PR #1360** (2026-04-04): Gaussian per-token NLL reweighting on PR #180 base → **+0.014 bpb (worse)**. Diagnosis: weighted train loss looks great but reshaped objective ≠ BPB metric.
- **PR #1233** (2026-04-01): Focal Loss (gamma=2.0) → **1.1460 bpb** (much worse than baseline at the time).
- **PR #1380** (2026-04-05): focal loss investigation, also negative.

What's different here:
1. **Stronger TTT-heavy base** (3-layer recurrence + parallel residuals + score-first TTT). Loss-shaping interacts with TTT, which itself adapts at eval time on hard tokens. The interaction is untested.
2. **Mean-1 normalization** keeps loss magnitude scale-stable across alpha — base optimizer LR doesn't need retuning.
3. **Detached weights with clipping** prevents single-token gradient blowup that #1360 suspected.
4. **Warmup ramp** (default 500 steps) avoids early-training collapse onto noisy NLL spikes.

If this mechanism is also negative on this stronger base, that's a stronger null than #1360/#1233 alone — three independent bases with the same negative direction is meaningful evidence.

## Base

- **PR #1413** (`bigbag`/`dexhunter`, 2026-04-09): SP8192 + 3-Layer Recurrence (L3-5) + Parallel Residuals (L7+) + QK-Gain 5.25 + Legal Score-First TTT
- 3-seed mean: **val_bpb 1.0810** (std 0.0002)
- Single change: `forward()` loss now supports per-token weighted CE controlled by `LOSS_REWEIGHT_ALPHA`

## Method

Per-token NLL rescaled by a detached, clipped, batch-mean-normalized weight:

```python
if alpha == 0.0:
    return F.cross_entropy(flat_logits, flat_targets, reduction="mean")  # bit-identical to PR #1413
nll = F.cross_entropy(flat_logits, flat_targets, reduction="none")
nll_d = nll.detach()
ratio = (nll_d / nll_d.mean().clamp(min=1e-8)).clamp(clip_lo, clip_hi)
ramp = clamp(self._train_step / warmup_steps, max=1.0)  # tensor scalar
a_eff = alpha * ramp
w = ratio.pow(a_eff)
w = w / w.mean().clamp(min=1e-8)  # mean-1 normalize → loss scale stable
return (w * nll).mean()
```

Properties:
- `alpha == 0.0`: constant Python branch returns original `F.cross_entropy(reduction="mean")`. Bit-identical to PR #1413 when disabled. `torch.compile(fullgraph=True)` resolves the branch at trace time.
- `alpha > 0`: upweight hard tokens (high NLL). Standard focal-style intuition.
- `alpha < 0`: upweight easy tokens (anti-focal). Sweep as control.
- Weights detached → no second-order gradient through weighting.
- Mean-1 normalization → loss scale stable across alpha → no LR retune needed.
- Warmup ramp → avoids early-step NLL spike collapse.
- Clip range [0.1, 10.0] → bounds single-token batch domination.
- `_train_step` is a non-persistent buffer set inside `step_fn` via `fill_(float(step))` before the model forward call. Buffer reads inside `forward` are torch ops (compile-safe).

## Why this could help BPB on this specific base

- BPB aggregates per-token NLL uniformly across val.
- Standard CE optimizer gradient is dominated by tokens with `loss × frequency` — converges on common easy tokens, stalls on long tail.
- Reweighting shifts gradient toward residual long-tail loss.
- This base has **TTT at eval time** — TTT itself adapts to hard val tokens. Two questions:
  1. Does train-time hardness focus *help* TTT (better starting point on long-tail manifold)?
  2. Or does it *hurt* by over-specializing the pre-TTT model on rare patterns that don't generalize?
- PR #1360's failure on a non-TTT base doesn't tell us which. This run does.

Risk: high-NLL tokens may be *inherently* unpredictable (rare names, numbers, non-English). Clip + warmup bound but don't eliminate. Sweep tells empirically.

## Hyperparameters

| Env var | Default | Sweep range |
|---|---|---|
| `LOSS_REWEIGHT_ALPHA` | `0.0` | `{0.0, 0.5, 1.0, 2.0, -0.5}` |
| `LOSS_REWEIGHT_CLIP_LO` | `0.1` | fixed |
| `LOSS_REWEIGHT_CLIP_HI` | `10.0` | fixed |
| `LOSS_REWEIGHT_WARMUP_STEPS` | `500` | fixed |

## Targets

- Base PR #1413 3-seed mean: **1.0810**
- Beats base (validates mechanism on TTT base): **< 1.0810** by > seed std (0.0002)
- Currently merged to main: **1.0810** (PR #1413 itself)
- Current open frontier (NN-only legal): **1.06157** (PR #1797, not merged to main)

This base + main alignment is record-track: a mechanism win here on the merged frontier is directly significant.

## Requirements

**Hardware**: 8× H100 SXM (FA3 Hopper-only).
**Python**: ≥ 3.12.
**Packages**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```
**Data**: plain SP8192 FineWeb shards in `./data/datasets/fineweb10B_sp8192/`.

## Reproduction

CWD = repo root. Pass script by full path.

Sanity (`alpha=0.0`, must reproduce PR #1413):
```bash
SCRIPT=records/track_10min_16mb/2026-04-25_LossReweight_on_3LayerRecur_ParResid/train_gpt.py
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  LOSS_REWEIGHT_ALPHA=0.0 RUN_ID=train_seed42_a0 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_a0.log 2>&1
```

Single seed at `alpha=1.0`:
```bash
SCRIPT=records/track_10min_16mb/2026-04-25_LossReweight_on_3LayerRecur_ParResid/train_gpt.py
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  LOSS_REWEIGHT_ALPHA=1.0 RUN_ID=train_seed42_a1 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_a1.log 2>&1
```

Sweep (single seed first; 3-seed `{0, 42, 1337}` only on winning alpha):
```bash
SCRIPT=records/track_10min_16mb/2026-04-25_LossReweight_on_3LayerRecur_ParResid/train_gpt.py
for a in 0.0 0.5 1.0 2.0 -0.5; do
  SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
    LOSS_REWEIGHT_ALPHA=$a RUN_ID=train_seed42_a${a} \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    > train_seed42_a${a}.log 2>&1
done
```

## Decision rules

- **alpha=0 sanity** must reproduce PR #1413 within seed noise — confirms patch is no-op when disabled.
- **best alpha** must beat 1.0810 by > seed std (0.0002) on a single seed → trigger 3-seed validation on `{0, 42, 1337}`.
- **3-seed mean** must beat 1.0810 by > 0.005 nat at p<0.01 to be a record.
- If no alpha beats base at single seed → mechanism is null on TTT bases too. Document as a notable non-record reinforcing #1360/#1233.

## Implementation

Patch is 4 small edits to PR #1413's `train_gpt.py` (LZMA-recompressed):
- Hyperparameters block: 4 new env vars (`LOSS_REWEIGHT_*`).
- `GPT.__init__`: stores hparams + registers `_train_step` non-persistent buffer.
- `GPT.forward`: branches on `alpha == 0.0`.
- `step_fn`: `base_model._train_step.fill_(float(step))` at start of each step.

Wrapped script size: 16,902 bytes (vs base 16,594 — +308 bytes). Tightest base seed has ~8KB headroom under 16,000,000-byte cap → fits.

## Results

_Pending — alpha=0 sanity run first, then sweep._

## Credits

- **PR #1413** base: `@bigbag`, `@dexhunter`
- **PR #1394** SP8192/GPTQ: `@clarkkev`
- **PR #1331/#1437** depth recurrence: `@dexhunter`
- **PR #1412/#1204** parallel residuals: `@Robby955`, `@msisovic`
- **PR #549/#1413** legal TTT: `@abaybektursun`, `@dexhunter`
- **PR #1360/#1233/#1380** prior loss-reweight negatives: motivating prior art.
