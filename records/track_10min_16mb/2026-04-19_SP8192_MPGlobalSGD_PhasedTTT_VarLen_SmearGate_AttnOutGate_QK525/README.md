# SP8192 + Multi-Phase Global SGD + Phased TTT + VarLen + SmearGate + AttnOutGate + QK-Gain 5.25

## Summary

Stage 2 of cross-stack port: adds **AttnOutGate** (per-head multiplicative gate inside attention) on top of Stage 1 (SmearGate + QK-Gain 5.25 over PR #1700's SP8192 + Multi-Phase Global SGD + Phased TTT + VarLen + DepthRec stack).

Together, this fully ports PR #1667's three contributions (SmearGate, AttnOutGate, QK-Gain 5.25) onto PR #1700's pipeline.

**Stage 1 result (seed 42):** val_bpb 1.07219 (eval 425s) — beat PR #1700 base by 0.00114, tied PR #1667 seed 42 (1.07221). Confirmed gates stack additively with PR #1700's TTT scheme. Not enough alone for record (threshold ≤ 1.06945 mean).

**Stage 2 hypothesis:** AttnOutGate adds another ~0.0010–0.0015 BPB on top of Stage 1, pushing toward ≤ 1.0710 (new legal frontier).

## Bases

- **Stage 1** (this repo, branch `SP8192-MPGlobalSGD-PhasedTTT-VarLen-SmearGate-QK525`, commit 792b00b): seed 42 = **1.07219**
- **PR #1667** (MarioPaerle, 2026-04-17): full PR1667 stack mean **1.07139**, seed 42 = 1.07221
- **PR #1700** (jorge-asenjo, 2026-04-18): full PR1700 stack mean **1.07219**, seed 42 = 1.07332

## Changes from Stage 1

| Parameter | Stage 1 | **Stage 2** |
|-----------|---------|-------------|
| `GATE_ATTN_OUT` | (n/a) | **1** (on by default) |
| `GATE_ATTN_SRC` | (n/a) | **proj** (gates attn-norm input) |
| `GATE_WIDTH` | (n/a) | **12** |

### AttnOutGate implementation

Per-head multiplicative gate applied to attention output before the output projection:

```python
y = flash_attn_3_func(q, k, v, causal=True)
if attn.use_xsa:
    y = attn._xsa_efficient(y, v)
if self.gate_attn_out:
    gate_in = x[..., : self.gate_width].contiguous()
    g = 2.0 * torch.sigmoid(self.attn_gate_proj(gate_in))
    y = y * g.unsqueeze(-1).to(dtype=y.dtype)
y = y.reshape(bsz, seqlen, dim)
```

Source `x` is the attn-norm input (`block.attn_norm(x_in) * block.ln_scale_factor`). Gate output: `[B, T, num_heads]`, broadcast across head_dim. Initial gate ≈ 1.0 (zero-init projection → sigmoid(0)=0.5 → ×2 = 1.0), so Stage 2 starts numerically identical to Stage 1.

**New params per layer:** `attn_gate_proj.weight` shape `[num_heads=8, gate_width=12]` = 96 floats. Across 12 layers = 1152 floats total. Auto-passthrough via PR #1700's `numel ≤ 65536` quant rule. Routed to scalar AdamW via `attn_gate_proj` entry in `CONTROL_TENSOR_NAME_PATTERNS`. CastedLinear gets `.float()` restoration via existing `restore_fp32_params`.

### Inline TTT path coverage

PR #1700 reimplements attention inline in two TTT paths: `_block_with_lora` (sequential layers) and `_parallel_block_with_lora` (parallel lanes after `parallel_start_layer`). Both paths now apply the same gate after `flash_attn_3_func` and `_xsa_efficient`, before the reshape, using `n` (the post-norm input) as gate source — keeping behavior identical between forward_logits and forward_ttt.

## Key metrics to compare

- Val BPB targets (seed 42):
  - ≤ 1.07219 = beats Stage 1 (validates AttnOutGate adds value here)
  - ≤ 1.07139 = beats PR #1667 mean (would be a new legal record)
  - ≤ 1.06945 = passes record threshold (≥0.005 nat improvement at p<0.01)
- Training steps in 600s (expect similar to Stage 1, ~4500-4800)
- Submission size < 16,000,000 bytes

## Reproduction

One-time setup:
```bash
pip install brotli sentencepiece python-minifier
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

Run (PR #1700's full env stack, gates default-on in this script):
```bash
SEED=42 RUN_ID=train_seed42 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-19_SP8192_MPGlobalSGD_PhasedTTT_VarLen_SmearGate_AttnOutGate_QK525/train_gpt.py
```

## Results

_Pending — run on 8xH100 SXM_
