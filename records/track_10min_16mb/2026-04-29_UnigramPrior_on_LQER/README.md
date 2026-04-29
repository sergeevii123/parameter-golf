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

This dir is self-contained: same aux files as PR #1855 (`lossless_caps.py`, `prepare_caseops_data.py`, `tokenizers/...`, `requirements.txt`). The `train_gpt.py` is PR #1855's with the unigram-prior patch applied.

### 0. Hardware + OS prerequisites

- 8× H100 80GB SXM (FA3 is Hopper-only; Ampere/Ada will fail at `from flash_attn_interface import flash_attn_func`).
- CUDA 12.8, Python ≥ 3.12.
- System binary `lrzip` (used by `COMPRESSOR=pergroup`):
  ```bash
  sudo apt-get install -y lrzip
  ```

### 1. Python deps

```bash
pip install -r records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/requirements.txt
# FlashAttention 3 (not on PyPI, install from prebuilt wheel):
pip install --no-deps flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
# verify FA3 imports cleanly:
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

### 2. Data prep (CaseOps tokenized FineWeb shards)

Two steps: download canonical doc stream from Hugging Face, then run CaseOps tokenizer + byte sidecar to produce the binary shards `train_gpt.py` reads.

```bash
# (a) Download docs_selected.jsonl from the official challenge HF dataset
#     (~6 GB raw text). MATCHED_FINEWEB_REPO_ID overrides the default repo.
python3 data/download_hf_docs_and_tokenize.py \
  --docs ./fineweb10B_raw/docs_selected.jsonl

# Alternative if you only need the doc stream (skips re-tokenization with the
# default SP8192 tokenizer): you can also use data/cached_challenge_fineweb.py
# with the --include-docs flag — both pull from willdepueoai/parameter-golf.

# (b) Tokenize with CaseOps SP model + emit per-token byte sidecar.
#     Writes to ./data/datasets/fineweb10B_sp8192_caseops/datasets/...
python3 records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/prepare_caseops_data.py \
  --docs ./fineweb10B_raw/docs_selected.jsonl \
  --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
  --sp   records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

After step (b) the layout under `./data/` is:

```
data/datasets/fineweb10B_sp8192_caseops/datasets/
  tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
  datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
    fineweb_train_000000.bin   # uint16 token shards
    fineweb_train_000001.bin
    ...
    fineweb_val_000000.bin
    fineweb_val_bytes_000000.bin   # parallel uint16 byte-count sidecar (canonical pre-CaseOps bytes)
```

The byte sidecar is **load-bearing** for legality: BPB is computed against original UTF-8 bytes, not transformed CaseOps tokens. Don't skip it.

Disk: ~19 GB after tokenization (raw docs + shards + sidecar). Make sure you're on a 50+ GB volume.

All run commands assume **CWD = repo root**. The hyperparameter overrides are PR #1855's defaults (greedy-validated 9-hparam stack) and must NOT change for valid comparison — only `UNIGRAM_PRIOR_*`, `SEED`, and `RUN_ID` differ across runs.

### 3. Sanity run (`UNIGRAM_PRIOR_ENABLED=0`, must reproduce PR #1855 seed 42 = 1.05989)

```bash
SCRIPT=records/track_10min_16mb/2026-04-29_UnigramPrior_on_LQER/train_gpt.py
DATA_DIR=./data \
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
SEED=42 \
UNIGRAM_PRIOR_ENABLED=0 RUN_ID=train_seed42_up0 \
torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_up0.log 2>&1
```

The full env block is verbose — for repeated runs put it in a bash array `COMMON_ENV` and dispatch via `env "${COMMON_ENV[@]}" SEED=$s ... torchrun ...`.

### 4. Single-shot test (`UNIGRAM_PRIOR_ENABLED=1`, seed 42)

Same env block, two flags flipped:

```bash
... (same env as step 3) ...
SEED=42 \
UNIGRAM_PRIOR_ENABLED=1 UNIGRAM_PRIOR_SHARDS=1 RUN_ID=train_seed42_up1 \
torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed42_up1.log 2>&1
```

### 5. 3-seed validation (only if seed-42 wins by ≥ 0.001 vs PR #1855 seed-42 = 1.05989)

```bash
for s in 42 0 1234; do
  ... (same env as step 3) ...
  SEED=$s \
  UNIGRAM_PRIOR_ENABLED=1 UNIGRAM_PRIOR_SHARDS=1 RUN_ID=train_seed${s}_up1 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" > train_seed${s}_up1.log 2>&1
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
