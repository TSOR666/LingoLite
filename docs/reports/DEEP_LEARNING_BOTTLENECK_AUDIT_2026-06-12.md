# Deep Learning Bottleneck Audit - 2026-06-12

## 1. Inferred Task and Model Family

- Intended task: multilingual neural machine translation.
- Model family: pre-norm Transformer encoder-decoder with RMSNorm, RoPE,
  grouped-query attention, SwiGLU, tied embeddings, autoregressive decoding,
  KV caching, and beam search.
- Inputs: source token IDs/masks `(B, S)` and target token IDs/masks `(B, T)`.
- Outputs: vocabulary logits `(B, T, V)` or generated token sequences.
- Training target: CUDA/CPU PyTorch; AMP, gradient accumulation, checkpointing,
  and `torch.compile` are optional.
- Inference target: CPU/mobile/server, with dynamic INT8 and ONNX paths.
- Quality metrics: BLEU, chrF/chrF++, perplexity, plus exact-match smoke tests.
- Deployment status: framework only. No real training corpus, validated
  checkpoint, or production quality baseline is shipped.

## 2. Critical Paths & Profiling Plan

Critical call graph:

`JSON pairs -> TranslationDataset.__getitem__ -> tokenizer.encode -> collate_fn
-> MobileTranslationModel.compute_loss -> encoder -> decoder -> tied LM head
-> cross entropy -> backward -> AdamW -> scheduler -> checkpoint`

Inference:

`tokenizer.batch_encode -> encoder -> cached decoder loop / beam loop
-> tokenizer.batch_decode -> BLEU/chrF or API response`

Audit execution:

1. Read architecture, data, training, generation, quantization, API, evaluation,
   and export paths.
2. Run the existing suite and synthetic shape/cache/training checks.
3. Reproduce checkpoint/export failures with non-default model dimensions.
4. Run tiny overfit training and all generation strategies.
5. Benchmark FP32 versus INT8 train/decode paths.
6. Profile CPU operators and estimate parameter/logit/LM-head costs.
7. Patch confirmed blocker/high issues and add regression coverage.

## 3. Bottleneck Summary

| severity | category | file:line | bottleneck | impact | proposed fix |
|---|---|---|---|---|---|
| BLOCKER (fixed) | Implementation | `lingolite/training.py:573`, `lingolite/mobile_translation_model.py:488` | Trainer checkpoints omitted architecture config, so tiny/custom models could not be reconstructed | Documented evaluation/export failed with shape mismatches | Store exact constructor config and use one strict loader |
| BLOCKER (fixed) | Inference | `scripts/export_mobile.py:19`, `scripts/export_mobile.py:190` | Mobile export ignored `model_state_dict`, used `strict=False`, and declared one output for a tuple-returning model | Could silently export random weights or fail output tracing | Strict shared loader plus logits-only wrapper |
| HIGH (fixed) | Inference | `lingolite/quantization_utils.py:218-289` | Saved INT8 state dicts lacked architecture/backend metadata | Quantized artifacts could not be reconstructed | Portable quantized checkpoint format and strict backend-aware loader |
| HIGH (fixed) | Memory | `lingolite/mobile_translation_model.py:289-361` | Full vocabulary logits were materialized for every target position | Default tiny, batch 32, T=127, V=24k: about 372 MiB FP32 logits alone | Training now projects bounded flattened-token chunks; default maximum is 1024 tokens |
| HIGH (fixed) | Speed | `lingolite/generation_utils.py:848-1080` | Beam search transferred candidates to CPU and rebuilt Python lists/tensors every token | GPU synchronization and poor scaling with batch/beams | EOS tracking, scores, live-beam selection, and finalization are tensorized on device |
| HIGH | Quality/Data | `README.md:13-17` | No real corpus, trained checkpoint, BLEU baseline, or domain validation | Architecture quality and deployment fitness are unproven | Train/evaluate on FLORES/OPUS/WMT-style held-out sets |
| MEDIUM | Speed | `lingolite/quantization_utils.py:99-103` | Compatibility INT8 dequantizes weights during every linear call | Smaller artifact but slower inference on audited CPU | Install/use `torchao` or target a true int8 runtime |
| MEDIUM (mitigated) | Architecture | `lingolite/mobile_translation_model.py:662-668` | Legacy presets use `d_ff=4*d_model` despite SwiGLU having three projections | About 50% more FFN parameters/MACs than compute-matched SwiGLU | New models can opt into checkpoint-safe `efficient_ffn=True` near `8/3*d_model` |
| MEDIUM (fixed) | Data | `lingolite/training.py:31-105` | SentencePiece tokenization ran in every `__getitem__`, every epoch | CPU/dataloader bottleneck on repeated corpora | `pretokenize=True` and `--pretokenize` encode each example once |
| MEDIUM (fixed) | Training | `lingolite/training.py:430-565` | `loss.item()` synchronized every micro-batch | Prevented fully asynchronous CUDA training | Main loop accumulates detached losses on device and synchronizes at log cadence |
| MEDIUM (fixed) | Memory | `lingolite/model_components.py:375-530`, `lingolite/generation_utils.py:860-1070` | Encoder states, masks, and cross K/V were physically repeated for every beam | Beam memory grew linearly with beam count | Cross-attention broadcasts source batch over beam groups and caches source K/V once |
| LOW (fixed) | Testing | `tests/test_cache_fix.py` | Several pytest tests returned booleans instead of asserting | False confidence and pytest warnings | Tests now raise assertions |

## 4. Detailed Findings

### A. Checkpoint and Export Correctness

- Evidence: trainer save logic at `lingolite/training.py:573-604` previously
  stored weights but no model config. Evaluation/export paths reconstructed a
  fixed small-like model. A custom 32-wide checkpoint reproduced a strict shape
  mismatch.
- Why it matters: a successfully trained model was not portable through the
  documented evaluation, API, quantization, or export workflows.
- Minimal fix: implemented exact config metadata, compiled-prefix normalization,
  strict state loading, and atomic writes.
- Stronger alternative: version the artifact schema and add migration tooling.
- Expected impact: correctness/deployment blocker removed; negligible runtime
  impact. Checkpoints gain only small metadata.

### B. Vocabulary Projection and Loss

- Evidence: `TransformerDecoder.lm_head` is a dense `d_model -> vocab_size`
  projection at `lingolite/encoder_decoder.py:365,475`. The default tiny model
  dedicates 6.144M/14.800M parameters (41.5%) to the shared embedding/head.
  Batch 32, T=127, V=24k creates about 372 MiB of FP32 logits and the head alone
  performs about 49.9 GFLOPs.
- Why it matters: this is a primary training-memory and decode-compute limit,
  independent of GQA/KV-cache efficiency.
- Implemented fix: decoder hidden-state computation is separated from the LM
  head, and training projects at most `logits_chunk_size` flattened target
  tokens at once. The default 1024-token chunk bounds FP32 logits to about
  93.75 MiB at V=24k instead of about 372 MiB for the cited batch.
- Stronger alternative: validated vocabulary reduction, target-language
  shortlists, adaptive softmax, or teacher-distilled smaller vocabularies.
- Expected impact: high memory reduction and moderate-to-high speed gain.
- Tradeoff: shortlist/adaptive methods complicate export and can hurt rare-word
  quality if not validated carefully.

### C. Beam Search Host Control

- Evidence: candidate tokens/scores are copied to CPU at
  `lingolite/generation_utils.py:966-968`, processed in Python, then copied back
  through new tensors at `1046-1057`.
- Why it matters: each generated token forces synchronization, defeating GPU
  queueing and reducing throughput at larger batch/beam counts.
- Implemented fix: top-k filtering, EOS handling, normalized finished scores,
  live-beam selection, and final selection remain on device. The loop performs
  no candidate `.cpu()` or `.tolist()` transfers.
- Stronger alternative: use a tested tensorized generation runtime and export a
  cache-aware decoder for the target backend.
- Expected impact: little CPU benefit at tiny sizes; substantial GPU/server
  latency improvement.
- Tradeoff: beam finalization semantics require careful regression tests.

### D. Beam Quality Is Not Guaranteed

- Evidence: the overfit smoke model produced 8/8 exact greedy/cached outputs but
  only 6/8 beam outputs. It shortened "buenos dias" to "dias" and "por favor" to
  "favor"; penalties from 0.8 through 2.0 did not change those results.
- Why it matters: the previous unconditional "+2-4 BLEU" claim was unsupported.
- Minimal fix: implemented documentation that requires validation-set tuning.
- Stronger alternative: coverage penalties, minimum length, n-gram blocking,
  calibration, or sequence-level knowledge distillation.
- Expected impact: potential quality gain, but only after corpus-level tuning.
- Tradeoff: constraints can harm valid short translations.

### E. INT8 Compatibility Backend

- Evidence: `DynamicQuantizedLinear.forward` converts INT8 weights/scales to the
  activation dtype on every call at `lingolite/quantization_utils.py:99-103`.
  Audited model size fell from 34.03 MiB to 9.66 MiB, but greedy/beam throughput
  fell from 319.6/375.3 to 302.1/244.1 tokens/s.
- Why it matters: it is a storage compatibility path, not a true optimized INT8
  kernel.
- Minimal fix: documentation now states the tradeoff and artifacts record the
  backend.
- Stronger alternative: deploy `torchao`, ONNX Runtime quantization, ExecuTorch,
  Core ML, or another target-specific integer kernel path.
- Expected impact: 3.52x smaller artifact now; real speedup only with a suitable
  backend.
- Tradeoff: backend-specific artifacts and quantization accuracy validation.

### F. Data and Quality Readiness

- Evidence: repository status explicitly says no checkpoint and no data at
  `README.md:13-17`. Only synthetic/tiny-corpus convergence is available.
- Why it matters: no evidence establishes BLEU, chrF, multilingual balance,
  hallucination behavior, long-sequence robustness, or domain transfer.
- Minimal fix: establish reproducible per-language baselines with fixed splits.
- Stronger alternative: temperature-balanced multilingual sampling, teacher
  distillation, domain adaptation, and human evaluation.
- Expected impact: this is the largest likely quality improvement.
- Tradeoff: requires data governance, compute, and sustained evaluation.

## 5. Patches Implemented

- Added exact model configuration to checkpoints.
- Added one strict checkpoint loader for trainer, raw, generic, compiled, and
  quantized artifacts.
- Made training and quantized checkpoint writes atomic.
- Fixed mobile export to load trainer weights and emit logits only.
- Wired API, evaluation, multilingual evaluation, ONNX export, and quantization
  through the shared loader.
- Added portable INT8/FP16 quantized checkpoint metadata and backend validation.
- Exposed gradient accumulation and gradient checkpointing in the training CLI.
- Added bounded chunked vocabulary loss with full-loss numerical parity.
- Tensorized beam candidate/finalization bookkeeping on device.
- Shared encoder states, source masks, and cross-attention K/V across beams.
- Added optional one-time dataset pretokenization.
- Deferred training loss synchronization to logging cadence.
- Added checkpoint-safe compute-matched SwiGLU sizing for new models.
- Converted boolean-returning pytest checks to assertions.
- Fixed benchmark decode throughput reporting.
- Removed unsupported beam-quality and static/QAT quantization claims.

Changed files include:

`lingolite/mobile_translation_model.py`, `lingolite/training.py`,
`lingolite/utils.py`, `lingolite/quantization_utils.py`,
`lingolite/generation_utils.py`, `scripts/export_mobile.py`,
`scripts/export_onnx.py`, `scripts/evaluate_model.py`,
`scripts/evaluate_multilingual.py`, `scripts/api_server.py`,
`scripts/benchmark.py`, `README.md`, and
`tests/test_checkpoint_portability.py` and
`tests/test_fit_for_purpose_optimizations.py`.

## 6. Tests Added + How to Run

Added `tests/test_checkpoint_portability.py` covering:

- custom architecture checkpoint reconstruction;
- compiled `_orig_mod.` state prefix normalization;
- malformed checkpoint rejection;
- logits-only mobile export wrapper;
- training CLI memory controls;
- decode throughput reporting;
- quantized checkpoint round trip;
- atomic-save failure behavior.

Added `tests/test_fit_for_purpose_optimizations.py` covering:

- chunked/full loss parity and bounded LM-head projection size;
- compute-matched SwiGLU parameter reduction;
- one-time dataset pretokenization;
- deferred trainer metric synchronization;
- absence of beam-search candidate CPU round trips;
- shared cross-attention K/V numerical parity and cache batch size.

Commands:

```bash
python -m pytest -q tests/test_checkpoint_portability.py
python -m pytest -q tests/test_model_components.py tests/test_encoder_decoder.py \
  tests/test_beam_search.py tests/test_cache_fix.py \
  tests/test_cached_decoder_wrapper.py tests/test_audit_regressions.py \
  tests/test_audit_optimizations.py tests/test_quantization_utils.py
python scripts/smoke_train.py --out-dir checkpoints/audit_smoke_20260612 \
  --max-steps 120 --batch-size 4 --seed 42
python scripts/smoke_infer.py --out-dir checkpoints/audit_smoke_20260612 \
  --threshold 1.0 --max-gen-length 20 --num-beams 3
```

Observed:

- Pre-patch full baseline: 308 passed, 3 skipped.
- Post-patch broad targeted set: 120 passed, 2 skipped.
- Additional focused post-patch set: 29 passed.
- Final broad bottleneck regression set: 154 passed, 2 skipped.
- Tiny training: loss 3.7642 -> 0.1420 in 74 steps.
- Tiny inference: greedy and cached 8/8 exact; cached/greedy agreement passed.
- CUDA and end-to-end ONNX runtime tests were unavailable in this CPU-only,
  no-ONNX environment.

## 7. Benchmark Results

Hardware/runtime: PyTorch 2.11.0 CPU build, 8 CPU threads, no CUDA.
Synthetic tiny preset, V=1000, B=2, S=T=16, decode length 16.

| variant | path | mean latency | throughput | model size |
|---|---:|---:|---:|---:|
| FP32 | train step | 81.85 ms | 366.5 target tok/s | 34.03 MiB |
| FP32 | greedy cached | 93.86 ms | 319.6 tok/s | 34.03 MiB |
| FP32 | beam-4 cached | 79.94 ms | 375.3 tok/s | 34.03 MiB |
| INT8 compatibility | greedy cached | 99.29 ms | 302.1 tok/s | 9.66 MiB |
| INT8 compatibility | beam-4 cached | 122.91 ms | 244.1 tok/s | 9.66 MiB |

Before/after latency was noisy because patches affect artifact/control paths,
not model kernels. The initial FP32 means were 99.34/74.80/87.79 ms for
train/greedy/beam; the final means were 81.85/93.86/79.94 ms. Treat these as
run-to-run variance, not a speed claim.

CPU profile, forward+backward: `aten::mm` was the largest self-time operator at
26.0%; dense linear algebra is the dominant compute class.

Peak CUDA memory was not measurable because CUDA is unavailable.

## 8. Optimization Roadmap

Immediate low-risk fixes:

1. Use the new self-describing checkpoint format everywhere.
2. Benchmark `torchao` INT8 on actual target CPUs; keep FP32 when it is faster.
3. Use `--pretokenize`, tuned workers, and pinned memory for repeated corpora.
4. Establish per-language BLEU/chrF and target-hardware acceptance thresholds.

Medium-risk architecture improvements:

1. Benchmark a fused cross-entropy kernel against the chunked fallback.
2. Evaluate `--efficient-ffn` quality before making it a default preset.
3. Distill from a stronger multilingual teacher and tune beam constraints.
4. Add target-language vocabulary shortlists only with rare-word evaluation.

High-risk/high-reward redesigns:

1. Distilled non-autoregressive or shallow-decoder translation for mobile.
2. Language-specific adapters/experts with a shared encoder.
3. Target-runtime graph optimization through ExecuTorch/ONNX Runtime/Core ML.
4. Vocabulary redesign and tokenizer retraining using real corpus statistics.

## 9. Final Verdict

**FIT-FOR-PURPOSE AS AN NMT ENGINEERING FRAMEWORK**

The previously identified implementation, checkpoint, export, vocabulary-loss,
beam-control, beam-memory, data-loading, and training-synchronization
bottlenecks now have tested code paths. Legacy checkpoint shapes remain
compatible, while new models can opt into a compute-matched SwiGLU width.

This is not yet evidence that any checkpoint is fit for production translation.
That decision remains gated on a real corpus, per-language BLEU/chrF and human
review, robustness testing, and latency/memory measurements on the target
runtime. The compatibility INT8 backend should still be treated as a storage
format unless a true integer-kernel backend demonstrates a measured speedup.
