# SP8192 + Multi-Phase Global SGD + Phased TTT + VarLen + SmearGate + AttnOutGate + QK-Gain 5.25 + **NoPE**

## Summary

Stage 3 of the cross-stack experiment: drops RoPE entirely (NoPE — no positional encoding) on top of the Stage 2 stack (PR #1700 base + PR #1667's three gates). FA3 has no `alibi_slopes` parameter (verified against `Dao-AILab/flash-attention/hopper/flash_attn_interface.py`), so ALiBi is blocked without a custom triton kernel; NoPE is the cheapest position-encoding axis to test under FA3.

**Stage 2 baseline (3-seed mean):** val_bpb 1.07117 (seeds {0, 42, 1337}, ties PR #1667 mean 1.07139, ties PR #1700 mean 1.07219).

**Stage 3 hypothesis:** Frontier stack (PR #1700 TTT + PR #1667 gates) is bottlenecked by absolute-position bias of RoPE under sliding-eval, as flagged by PR #1718 (*"the architecturally correct next step is relative-position attention where scored-token position becomes irrelevant"*). TTT phases already shuffle context positions during eval. Removing RoPE may free the model from a ceiling none of the existing PRs has touched.

**Risk:** Pure NoPE on causal LMs at 8K context can regress short-context BPB (Kazemnejad et al). If Stage 3 regresses badly, Stage 4 builds custom ALiBi triton kernel.

## Bases

- **Stage 2** (this repo, branch `SP8192-MPGlobalSGD-PhasedTTT-VarLen-SmearGate-AttnOutGate-QK525`, commit a8aa39c): 3-seed mean **1.07117**
- **PR #1667** (MarioPaerle, 2026-04-17): 3-seed mean **1.07139**
- **PR #1700** (jorge-asenjo, 2026-04-18): 3-seed mean **1.07219**

## Changes from Stage 2

| Parameter | Stage 2 | **Stage 3** |
|-----------|---------|-------------|
| `NOPE` | (n/a) | **1** (on by default) |
| `ROPE_DIMS` | 16 | 16 (unused under NoPE) |
| `ROPE_BASE` | 1e4 | 1e4 (unused under NoPE) |

### NoPE implementation

`Hyperparameters.nope` env knob (default `1`) propagates through `Block` → `CausalSelfAttention.nope`. When `nope` is true, the three attention paths skip `apply_rotary_emb` entirely:

```python
if not self.nope:
    cos, sin = self.rotary(seqlen, x.device, q.dtype)
    q = apply_rotary_emb(q, cos, sin, self.rope_dims)
    k = apply_rotary_emb(k, cos, sin, self.rope_dims)
q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
```

Same gate pattern in `_block_with_lora` and `_parallel_block_with_lora`. The `Rotary` module is still constructed (warmup calls remain harmless and idempotent) — keeps the diff minimal and reversible by flipping `NOPE=0`.

**No new params.** Causal mask remains via `flash_attn_3_func(causal=True)`. Order-sensitivity is preserved by the causal mask + LeakyReLU² MLP, but absolute and relative position information is dropped from attention.

### Why FA3 forces this rather than ALiBi

`flash_attn_3._flash_attn_forward` signature accepts `softcap`, `attention_chunk`, `window_size_left/right`, but no `alibi_slopes` (FA2 had it; FA3 dropped it). Three options were considered:

1. **NoPE** (this stage): zero kernel work, free under FA3.
2. **ALiBi via FA2**: ~3× slower attention, breaks the 600s budget with TTT.
3. **ALiBi via custom triton kernel**: 2-3 days of work; deferred to Stage 4 if NoPE proves the position axis matters.

## Key metrics to compare

- Val BPB targets (seed 42):
  - ≤ 1.07117 = beats Stage 2 (validates NoPE survives at frontier)
  - ≤ 1.07139 = beats PR #1667 mean
  - ≤ 1.06945 = passes record threshold (≥0.005 nat improvement at p<0.01)
- Training steps in 600s (expect ~+50 over Stage 2 from skipping rotary in three paths)
- Submission size < 16,000,000 bytes (no new params → unchanged)

## Reproduction

One-time setup:
```bash
pip install brotli sentencepiece python-minifier
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

Run (PR #1700's full env stack, NoPE default-on in this script):
```bash
SEED=42 RUN_ID=train_seed42 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-19_SP8192_MPGlobalSGD_PhasedTTT_VarLen_SmearGate_AttnOutGate_QK525_NoPE/train_gpt.py
```

To A/B against Stage 2 (RoPE on) using the same script:
```bash
NOPE=0 SEED=42 RUN_ID=train_seed42_rope ...  # reproduces Stage 2 numerics
```

## Results

### Seed 42 (run on 8xH100 SXM)

_Pending._

### Multi-seed (3-seed mean, seeds {0, 42, 1337})

_Pending — only run if seed 42 ≤ 1.07117 (otherwise NoPE doesn't beat Stage 2 baseline)._
