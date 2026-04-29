# Unigram-prior output bias on PR #1855 LQER base

**Status:** pending runs. Mechanism: add a fixed `(vocab_size,)` bias to output logits, initialized to `log p(token)` from training data. Disabled = bit-identical to PR #1855 (3-seed mean **1.06108**, current main frontier).

## Hypothesis

The lm_head (tied to embeddings) maps hidden state to logits with no additive bias. At init, predictions are roughly uniform over the 8192-token vocab (~13 nats/token). Standard CE training quickly learns the unigram distribution as a byproduct — gradient is spent on this trivial task during early steps.

If we **warm-start** the output bias to `log p(token)` from corpus statistics, the model's predictions at step 0 are already the unigram distribution (~7-8 nats/token, the corpus unigram entropy). Gradient capacity is freed from step 1 to learn higher-order structure that BPB integrates over.

Under a fixed 600s budget, freeing ~400-500 steps of "learn the unigram" overhead could shift the final BPB by 0.001–0.005 nat. If signal is real, this is a record.

## Method

```python
# In _project_logits (forward + forward_logits paths):
proj = F.linear(hidden, self.tok_emb.weight)  # tied
if self.unigram_prior_enabled:
    proj = proj + self.unigram_logits.to(dtype=proj.dtype)  # fixed log p(token)
return proj  # softcap+tanh applied downstream

# In forward_ttt (TTT eval path):
logits = F.linear(x, self.tok_emb.weight) + lora.lm_head_lora(x)
if self.unigram_prior_enabled:
    logits = logits + self.unigram_logits.to(dtype=logits.dtype)
logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
```

`unigram_logits` is a **non-trainable persistent buffer** of shape `(vocab_size,) = (8192,)`. Computed once in `train_model()` before `torch.compile`:

```python
counts = torch.zeros(vocab_size, dtype=torch.float64)
for f in first_N_train_shards:
    tokens = load_data_shard(f).long()
    counts += torch.bincount(tokens, minlength=vocab_size).double()
counts.add_(1.0)  # Laplace smoothing
log_p = (counts.log() - counts.sum().log()).float()
```

`UNIGRAM_PRIOR_SHARDS=1` (default) reads ~10M tokens from the first CaseOps train shard — enough for accurate unigram estimates at vocab=8192. Each token has hundreds of expected occurrences.

## Properties

- `UNIGRAM_PRIOR_ENABLED=0` → constant Python branch. `torch.compile(fullgraph=True)` resolves at trace time. Bit-identical to PR #1855.
- `UNIGRAM_PRIOR_ENABLED=1` → buffer added to logits before softcap. Affects all three forward paths (training, eval forward_logits, TTT).
- Buffer is **non-trainable**. v1 tests "does fixed unigram prior help?" If yes, v2 makes it a learnable parameter.
- Buffer is **persistent in state_dict** → saved/loaded across train/quant/eval/TTT. GPTQ pipeline routes it via `numel ≤ 65536` fp16 passthrough.
- Buffer **bypasses LQER, GPTQ, gate-int8** routing because its `ndim=1` (only 2D weight matrices hit those paths).
- TTT does not adapt the buffer (it's a buffer, not parameter; LoRA does not touch it).

## Cost

- **Code growth:** ~+800 bytes raw source. After pyminify + brotli-11 in submission, ~+50-100 bytes.
- **Buffer storage:** 8192 × 2 bytes (fp16 passthrough) = 16,384 bytes raw. Smooth log-Zipfian distribution → ~3-5 KB after brotli-11.
- **Total artifact growth:** ~5 KB worst case. PR #1855 tightest seed has 92,450 bytes headroom (`16,000,000 - 15,907,550`). Comfortable fit.
- **Compute overhead:** one `bincount` over ~10M tokens at startup (~1 second). Forward path adds one `+ self.unigram_logits` (8192-vector broadcast over `B*T` positions) per call. Negligible vs the rest of the forward pass.

## Reproduction

This dir contains the same auxiliary files as PR #1855 (`lossless_caps.py`, `prepare_caseops_data.py`, `tokenizers/...`, `requirements.txt`). The `train_gpt.py` is PR #1855's with the unigram-prior patch applied.

CWD = repo root. CaseOps shards must be tokenized first via PR #1855's pipeline (see `prepare_caseops_data.py`).

Sanity (`UNIGRAM_PRIOR_ENABLED=0`, must reproduce PR #1855 seed 42 = 1.05989):
```bash
SCRIPT=records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/train_gpt.py
SEED=42 CASEOPS_ENABLED=1 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  UNIGRAM_PRIOR_ENABLED=0 RUN_ID=train_seed42_up0 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_up0.log 2>&1
```

Single-shot test (`UNIGRAM_PRIOR_ENABLED=1`, seed 42):
```bash
SCRIPT=records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/train_gpt.py
SEED=42 CASEOPS_ENABLED=1 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  UNIGRAM_PRIOR_ENABLED=1 UNIGRAM_PRIOR_SHARDS=1 RUN_ID=train_seed42_up1 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_up1.log 2>&1
```

3-seed validation (only if seed-42 single shot wins by ≥ 0.001 BPB vs PR #1855 seed-42 = 1.05989):
```bash
SCRIPT=records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/train_gpt.py
for s in 42 0 1234; do
  SEED=$s CASEOPS_ENABLED=1 \
    DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    UNIGRAM_PRIOR_ENABLED=1 UNIGRAM_PRIOR_SHARDS=1 RUN_ID=train_seed${s}_up1 \
    torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    > train_seed${s}_up1.log 2>&1
done
```

## Targets

- PR #1855 seed-42 reference: **1.05989**
- Sanity (UNIGRAM_PRIOR_ENABLED=0) must match PR #1855 seed-42 to within ~0.0007 (3-seed std × 3) to confirm patch is no-op when disabled.
- Single-seed signal threshold (run 3-seed): **≤ 1.05889** (-0.001 nat).
- 3-seed record threshold: 3-seed mean ≤ 1.05608 (-0.005 nat below PR #1855 1.06108) at p<0.01.

## Decision rules

- **Sanity must reproduce PR #1855 seed-42** within seed noise → confirms patch is bit-equivalent when disabled. If it diverges meaningfully, stop and debug `_project_logits` integration with fused CE / LQER pipeline.
- **Single-shot ≥ 1.05889** (i.e. ≤ 0.001 nat improvement) → null. Stop. Document.
- **Single-shot in [1.05889, 1.05789]** (-0.001 to -0.002) → marginal. Run 3 seeds to disambiguate noise.
- **Single-shot ≤ 1.05789** (≥ -0.002) → strong signal. Run 3 seeds for record validation.

## Risks

- The model may already learn unigram statistics within the first 50-100 steps so quickly that the prior gives no measurable headroom by step 4900. Documented prior cost in classical NLP for transformers is ~few hundred steps; in our 4900-step regime, the asymptote may be unmoved.
- If the prior shifts logit *magnitudes* into a regime where logit_softcap=30 saturates differently, training dynamics could shift in unexpected ways. Mitigation: bias is added pre-softcap, so softcap+tanh squashes the same way the rest of the model does.
- LQER int4 + per-group compressor has not been tested with extra fp16 passthrough buffers. Pipeline should handle it (numel ≤ 65536 branch), but a ~5KB artifact growth could push tight seeds near the cap. Mitigation: PR #1855's worst seed has 92KB headroom — comfortable.

## Implementation

4 surgical edits to PR #1855's `train_gpt.py`:
1. **Hyperparameters block:** +2 env vars (`UNIGRAM_PRIOR_ENABLED`, `UNIGRAM_PRIOR_SHARDS`).
2. **GPT.__init__:** register fp32 persistent buffer `unigram_logits` of shape `(vocab_size,)` when enabled.
3. **GPT._project_logits:** add buffer to projected logits before return (covers training forward + eval `forward_logits`).
4. **GPT.forward_ttt:** add buffer to inline logits (post-LoRA, pre-softcap).
5. **train_model:** new helper `_compute_unigram_logits(h, device)` reads first N shards, bincounts, Laplace-smooths, copies log-prob into buffer. Called before `torch.compile`.

Total source delta: ~45 lines added. Disabled path is constant-folded by `torch.compile(fullgraph=True)` → zero runtime cost when off.

## Credits

- **PR #1855** base: `@codemath3000`
- **PR #1797** LQER: `@dexhunter`
- **PR #1787** Polar Express NS / MIN_LR / sparse gate / fused CE: `@nprime06`
- **PR #1767** TTT improvements: `@dexhunter`
- **PR #1729** CaseOps tokenizer: `@romeerp`
- **PR #1667** SmearGate: `@MarioPaerle`
