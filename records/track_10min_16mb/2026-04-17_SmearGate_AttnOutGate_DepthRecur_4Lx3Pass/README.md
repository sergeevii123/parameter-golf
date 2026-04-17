# SmearGate + AttnOutGate + Depth Recurrence 4L x 3 Passes

## Summary

Less aggressive depth recurrence: 4-layer × 3-pass loop (19 virtual layers) stacked on top of PR #1667 (1.0714 BPB). Matches PR #1678's virtual-layer count (loop_end=6 × 3 passes) but on the stronger gated base.

**Base:** PR #1667 (SmearGate + AttnOutGate + legal TTT, 1.0714 BPB).

**Prior experiment** (2026-04-17_SmearGate_AttnOutGate_DepthRecur_4Lx4Pass): 4L × 4 passes (23 virt) landed at 1.07306 — slightly worse than base. Step loss (-13%) ate capacity gain.

This variant trades less compute for smaller step penalty.

## Changes from base

| Parameter | PR #1667 | 4Lx4Pass (prior) | **This (4Lx3Pass)** |
|-----------|----------|------------------|---------------------|
| `LOOP_START` | 3 | 3 | 3 |
| `LOOP_END` | 5 | 6 | **6** |
| `NUM_LOOPS` | 2 | 3 | **2** |
| Virtual layers | 17 | 23 | **19** |
| Compute ratio | 1.00x | 1.35x | **1.12x** |
| Est. steps | 4840 | 4202 | **~4330** |

Gate defaults flipped on (match PR #1667 record): `SMEAR_GATE=1`, `GATE_ATTN_OUT=1`, `QK_GAIN_INIT=5.25`.

### Virtual layer layout

```
[0,1,2, 3,4,5,6, 3,4,5,6, 3,4,5,6, 7,8,9,10]
 prefix └─pass1─┘ └─pass2─┘ └─pass3─┘ suffix
```

- Encoder: `[0,1,2, 3,4,5,6, 3,4]` (9 virt)
- Decoder: `[5,6, 3,4,5,6, 7,8,9,10]` (10 virt)
- Skip weights: 9 (vs PR #1667's 8)

## Hypothesis

Promotes layer 6 into the loop (was in post-loop section) — adds one more unique layer to the recurrence core. +12% compute/step is mild; step loss (-10%) should be smaller than 4Lx4Pass's -13%.

If the monotonic "more virt = better" trend holds, 19 virt on gated base should beat PR #1667's 17 virt. Prior data:
- 17 virt, no gates → 1.0856 (SOTA no-TTT)
- 17 virt, with gates + TTT → 1.0714 (PR #1667)
- 19 virt, no gates → pending (PR #1678)
- 19 virt, with gates + TTT → **this PR**

## Key metrics to compare

- Val BPB targets (seed 42):
  - ≤ 1.07221 = beats PR #1667 base seed 42
  - ≤ 1.07306 = beats our prior 4Lx4Pass seed 42
- Training steps in 600s (expect ~4330)
- Submission size < 16MB

## Reproduction

One-time setup:
```bash
pip install brotli sentencepiece python-minifier
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

Run:
```bash
SEED=42 RUN_ID=train_seed42 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-17_SmearGate_AttnOutGate_DepthRecur_4Lx3Pass/train_gpt.py
```

## Results

_Pending — run on 8xH100 SXM_
