# SmearGate + AttnOutGate + Asymmetric Depth Recurrence

## Summary

Asymmetric shrinking-loop depth recurrence on top of PR #1667 (1.0714 BPB). Each successive pass drops the deepest looped layer: `[3,4,5,6] → [3,4,5] → [3,4]`. Same total looped compute as PR #1667 (9 layer-passes), but breaks the uniform-repeat symmetry so LoRA TTT sees distinguishable gradient paths per layer.

**Base:** PR #1667 (SmearGate + AttnOutGate + legal LoRA TTT, 1.0714 BPB).

**Prior experiments on this base:**
- 4Lx4Pass (23 virt, uniform) → 1.07306 (−) PR-branch `SmearGate-AttnOutGate-DepthRecur-4Lx4Pass`
- 4Lx3Pass (19 virt, uniform) → killed early at quant 1.08402 (projected ~1.074, −)

Both null/negative. Uniform depth recurrence appears saturated on this base — adding passes hurts step count more than virt-count helps. This experiment holds compute constant and perturbs only the loop pattern.

## Changes from base

| Parameter | PR #1667 | 4Lx4Pass (prior) | **This (Asymmetric)** |
|-----------|----------|------------------|-----------------------|
| `LOOP_START` | 3 | 3 | 3 |
| `LOOP_END` | 5 | 6 | **6** |
| `NUM_LOOPS` | 2 | 3 | **2** |
| `ASYMMETRIC_LOOP` | 0 | 0 | **1** |
| Looped-layer passes | 3+3+3 = 9 | 4×4 = 16 | **4+3+2 = 9** |
| Virtual layers | 17 | 23 | **16** |
| Compute ratio | 1.00x | ~1.35x | **~1.00x** |
| Est. steps | 4840 | 4202 | **~4840** |

Gate defaults match PR #1667 record: `SMEAR_GATE=1`, `GATE_ATTN_OUT=1`, `QK_GAIN_INIT=5.25`.

### Virtual layer layout

```
[0,1,2, 3,4,5,6, 3,4,5, 3,4, 7,8,9,10]
 prefix └─pass1─┘└─pass2┘└pass3┘ suffix
```

- Encoder: `[0,1,2, 3,4,5,6, 3]` (8 virt)
- Decoder: `[4,5, 3,4, 7,8,9,10]` (8 virt)
- Skip weights: 8 (vs PR #1667's 8)

### Per-layer pass counts

| Physical layer | Uniform 3-pass (PR #1667) | **Asymmetric** |
|---:|:---:|:---:|
| 3 | 3 | **3** |
| 4 | 3 | **3** |
| 5 | 3 | **2** |
| 6 | — (in suffix) | **1** |

Layer 6 is promoted into the loop at 1 pass; layer 5 demoted to 2 passes; layers 3,4 unchanged. Total loop touches = 9, same as PR #1667.

## Hypothesis

Uniform loops (all layers × N passes) give LoRA TTT a symmetric gradient: the same weight matrix is hit N times per forward, so TTT updates can't specialize across passes. Asymmetric shrinking breaks this:

- Layer 3,4 still get 3 touches → early-iterate refinement
- Layer 5 gets 2 → mid-iterate
- Layer 6 gets 1 → late-iterate

Each physical layer now sees a **distinct pass-count regime**, so its LoRA adapter trains against a gradient signal that is not a multi-count summation over identical paths. If PR #1667's TTT uplift was being smeared by the uniform-loop repetition, this pattern should recover some of that capacity.

Same compute = same step count. If BPB improves at matched steps, the gain is purely architectural.

## Key metrics to compare

- Val BPB targets (seed 42):
  - ≤ 1.07221 = beats PR #1667 base seed 42 (primary goal)
  - ≤ 1.07306 = beats prior 4Lx4Pass seed 42
- Training steps in 600s (expect ~4840)
- Submission size < 16,000,000 bytes (decimal MB)

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
  records/track_10min_16mb/2026-04-17_SmearGate_AttnOutGate_DepthRecur_Asymmetric/train_gpt.py
```

## Results

_Pending — run on 8xH100 SXM_
