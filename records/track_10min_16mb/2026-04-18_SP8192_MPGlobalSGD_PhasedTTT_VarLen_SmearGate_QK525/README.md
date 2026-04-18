# SP8192 + Multi-Phase Global SGD + Phased TTT + VarLen + SmearGate + QK-Gain 5.25

## Summary

Stage 1 of cross-stack port: lifts SmearGate (modded-nanogpt @classiclarryd) and QK-Gain 5.25 from PR #1667 onto the PR #1700 base (SP8192 + Multi-Phase Global SGD + Phased TTT + VarLen flash attention + fused triton MLP + 3-layer depth recurrence).

**Why Stage 1 only:** PR #1667 contributes three things (SmearGate, AttnOutGate, QK-Gain 5.25 + LoRA TTT). SmearGate + QK-Gain are model-level / scalar changes — easy to port without touching PR #1700's weight-bank attention infrastructure. AttnOutGate sits *inside* the attention forward and would need bank/serialization surgery — deferred to Stage 2 if Stage 1 is positive.

## Bases

- **PR #1667** (MarioPaerle, 2026-04-17): SmearGate + AttnOutGate + Legal LoRA TTT, **mean val_bpb 1.07139** (current legal frontier, seed 42 = 1.07221)
- **PR #1700** (jorge-asenjo, 2026-04-18): SP8192 + Multi-Phase Global SGD + Phased TTT + VarLen + DepthRec, **mean val_bpb 1.07219** (seed 42 = 1.07332)

The two stacks are largely orthogonal:
- PR #1700 = better TTT scheme (multi-phase global SGD on top of phased LoRA TTT) + VarLen + fused MLP
- PR #1667 = better base architecture (gates) + cleaner LoRA TTT

If gates stack additively, expected target ≤ 1.069–1.070.

## Changes from PR #1700 base

| Parameter | PR #1700 base | **This PR** |
|-----------|---------------|-------------|
| `QK_GAIN_INIT` | 5.0 | **5.25** |
| `SMEAR_GATE` | (n/a) | **1** (on by default) |
| `SMEAR_GATE_WIDTH` | (n/a) | **12** |

### SmearGate implementation

Inserted between `tok_emb` and `rms_norm` in both `forward_logits` and `forward_ttt`:

```python
x = self.tok_emb(input_ids)
if self.smear_gate_enabled:
    sl = self.smear_lambda.to(dtype=x.dtype)
    g = sl * torch.sigmoid(self.smear_gate(x[:, 1:, : self.smear_width]))
    x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
x = F.rms_norm(x, (x.size(-1),))
```

Token at position t (for t≥1) becomes `x[t] + g_t * x[t-1]`. Token 0 unchanged. Init: `smear_gate.weight=0`, `smear_lambda=0` → fully transparent at init.

**New params:** `smear_gate.weight` (1×12 = 12 floats), `smear_lambda` (1 float). Auto-passthrough via PR #1700's `numel ≤ 65536` quant rule. Registered with the scalar AdamW optimizer alongside `skip_weights`, `parallel_*_lambdas`. CastedLinear gets float() restoration via existing `restore_fp32_params` path. No serialization changes needed.

## What's deferred to Stage 2

- **AttnOutGate** (`GATE_ATTN_OUT=1`): per-layer per-head multiplicative gate inside attention forward. Requires plugging into PR #1700's `CausalSelfAttention.forward` between `flash_attn` output and `out_proj`. ~1056 new params per model — needs bank/serialization integration.

If Stage 1 lands ≤ 1.0710, Stage 2 is worth pursuing. If Stage 1 regresses, the gates may be stack-incompatible with PR #1700's TTT scheme — pivot to other axes.

## Key metrics to compare

- Val BPB targets (seed 42):
  - ≤ 1.07332 = beats PR #1700 seed 42 (validates the port didn't break anything)
  - ≤ 1.07221 = beats PR #1667 seed 42 (would be a new legal record)
- Training steps in 600s (expect similar to PR #1700, ~4500-4800)
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
  records/track_10min_16mb/2026-04-18_SP8192_MPGlobalSGD_PhasedTTT_VarLen_SmearGate_QK525/train_gpt.py
```

## Results

_Pending — run on 8xH100 SXM_
