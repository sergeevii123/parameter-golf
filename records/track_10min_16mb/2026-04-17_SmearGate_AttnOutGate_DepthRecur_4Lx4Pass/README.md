# SmearGate + AttnOutGate + Depth Recurrence 4L x 4 Passes

## Summary

Stacks 4-layer × 4-pass depth recurrence (23 virtual layers) on top of PR #1667's SmearGate + Attention Output Gate + legal TTT recipe (1.0714 BPB).

**Base:** PR #1667 (SmearGate + AttnOutGate + legal TTT, 1.0714 BPB). Best current legal non-casefold non-VarLen result.

## Changes from base

| Parameter | PR #1667 | This experiment |
|-----------|----------|-----------------|
| `LOOP_START` | 3 | 3 |
| `LOOP_END` | 5 | 6 |
| `NUM_LOOPS` | 2 | 3 |

Gates (inherited from PR #1667, defaults flipped so no env vars needed):
- `SMEAR_GATE=1` (was 0 default, on at record)
- `GATE_ATTN_OUT=1` (was 0 default, on at record)
- `QK_GAIN_INIT=5.25` (was 5.0 default, 5.25 at record)

### Virtual layer layout

**PR #1667 base (17 virt):**
```
[0,1,2, 3,4,5, 3,4,5, 3,4,5, 6,7,8,9,10]
```

**This PR (23 virt):**
```
[0,1,2, 3,4,5,6, 3,4,5,6, 3,4,5,6, 3,4,5,6, 7,8,9,10]
         └──── 4 passes through 4 layers ────┘
```

- Encoder: `[0,1,2, 3,4,5,6, 3,4,5,6]` (11 virt)
- Decoder: `[3,4,5,6, 3,4,5,6, 7,8,9,10]` (12 virt)
- Skip weights: 11 (vs PR #1667's 8)

## Hypothesis

PR #1667 gains ~0.009 BPB over PR #1583 via two zero-initialized gates (SmearGate + AttnOutGate) + legal TTT. These are orthogonal to depth recurrence. Stacking deeper recurrence on top should compose.

Prior data (from PR #1678's table) shows monotonic improvement with virtual depth:
- 17 virt → 1.0856 (SOTA no-TTT)
- 19 virt → pending (PR #1678, loop_end=6 × 3 passes)

This PR tests 23 virt (loop_end=6 × 4 passes) on the stronger gated base.

## Compute tradeoff

| Config | Virt layers | Compute/step ratio | Est. steps in 600s |
|--------|------------|---------------------|---------------------|
| PR #1667 | 17 | 1.00x | ~4,840 (measured) |
| This PR | 23 | ~1.35x | ~3,580 |

Step reduction: -26%. Quality-vs-steps tradeoff same as prior depth-recurrence experiments.

## Key metrics to compare

- Val BPB targets (3-seed mean):
  - ≤ 1.0801 = beats base SOTA (PR #1583)
  - ≤ 1.0788 = beats trajectory-state readout (PR #1676)
  - ≤ 1.0714 = new legal SOTA (must beat PR #1667 base)
- Training steps completed in 600s (expect ~3,580, vs PR #1667's 4,840)
- Quantized model size < 16MB (physical layers unchanged)

## Reproduction

One-time setup:
```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

Run (defaults already match PR #1667's record recipe + our loop change; no env vars required):
```bash
SEED=42 RUN_ID=train_seed42 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-17_SmearGate_AttnOutGate_DepthRecur_4Lx4Pass/train_gpt.py
```

## Results

_Pending — run on 8xH100 SXM_
