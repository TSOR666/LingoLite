"""Performance benchmark for LingoLite.

Synthetic-input benchmark covering the three hot paths:

  * train_step  - forward + backward + optimizer step on a random batch
  * greedy      - autoregressive decode via the KV-cache fast path
  * beam        - beam-search decode

For each scenario we report wall-clock latency (mean, p50, p95), throughput
(tokens/sec), and (on CUDA) peak allocated memory. A short warmup is run
before timed iterations so JIT/cudnn/allocator effects don't pollute the
first sample.

With ``--variants``, each scenario is run across multiple model variants
(e.g. fp32 vs int8 dynamic quantization) and the matrix is reported in a
single table along with the model's serialized size. ``train_step`` is
silently skipped for variants that don't support training (int8).

The script does **not** require a trained checkpoint; weights are randomly
initialized via ``create_model``. The goal is to measure the implementation's
performance characteristics, not translation quality.

Usage examples:

    # Default: small model, all three scenarios, CPU or CUDA if available
    python scripts/benchmark.py

    # Just greedy decoding on a tiny model with a longer decode budget
    python scripts/benchmark.py --model-size tiny --scenarios greedy --max-length 64

    # Compare fp32 vs int8 dynamic quantization for inference latency + size
    python scripts/benchmark.py --variants fp32 int8 --scenarios greedy beam

    # JSON output for machine consumption / regression tracking
    python scripts/benchmark.py --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch

from lingolite.mobile_translation_model import MobileTranslationModel, create_model


SCENARIOS = ("train_step", "greedy", "beam")
VARIANTS = ("fp32", "fp16", "bf16", "int8")
# int8 (dynamic quantization) replaces nn.Linear modules in-place with custom
# eager-mode wrappers; gradients aren't supported, so train_step is skipped.
INFERENCE_ONLY_VARIANTS = {"int8"}


@dataclass
class ScenarioResult:
    name: str
    iters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    tokens_per_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    variant: str = "fp32"
    model_size_mb: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(requested)


@contextmanager
def _cuda_memory_scope(device: torch.device) -> Iterator[Callable[[], Optional[float]]]:
    """Reset and report peak allocated memory in MB on CUDA; no-op on CPU."""
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    captured: Dict[str, Optional[float]] = {"peak_mb": None}

    def report() -> Optional[float]:
        return captured["peak_mb"]

    try:
        yield report
    finally:
        if is_cuda:
            torch.cuda.synchronize(device)
            captured["peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_iters(
    fn: Callable[[], int],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[List[float], int]:
    """Run ``fn`` ``warmup`` then ``iters`` times. ``fn`` returns tokens processed.

    Returns (per-iter wall-clock seconds, last tokens-processed).
    Each call is bracketed by a device sync so timings reflect actual completion
    rather than just kernel-launch latency.
    """
    last_tokens = 0
    for _ in range(warmup):
        last_tokens = fn()
        _sync(device)

    timings: List[float] = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        last_tokens = fn()
        _sync(device)
        timings.append(time.perf_counter() - t0)
    return timings, last_tokens


def _summarize(
    name: str,
    timings_s: List[float],
    *,
    tokens_per_call: Optional[int],
    peak_memory_mb: Optional[float],
    variant: str = "fp32",
    model_size_mb: Optional[float] = None,
    notes: Optional[List[str]] = None,
) -> ScenarioResult:
    timings_ms = [t * 1000.0 for t in timings_s]
    sorted_ms = sorted(timings_ms)
    n = len(sorted_ms)
    p50 = sorted_ms[n // 2]
    # p95 = highest sample below the 95th percentile boundary; clamp index.
    p95 = sorted_ms[min(n - 1, max(0, int(round(0.95 * (n - 1)))))]
    mean = statistics.fmean(timings_ms)

    tps: Optional[float] = None
    if tokens_per_call is not None and tokens_per_call > 0 and mean > 0:
        tps = tokens_per_call * 1000.0 / mean  # tokens / sec

    return ScenarioResult(
        name=name,
        iters=n,
        mean_ms=mean,
        p50_ms=p50,
        p95_ms=p95,
        min_ms=min(timings_ms),
        max_ms=max(timings_ms),
        tokens_per_sec=tps,
        peak_memory_mb=peak_memory_mb,
        variant=variant,
        model_size_mb=model_size_mb,
        notes=list(notes or []),
    )


def _measure_model_size_mb(model: torch.nn.Module) -> float:
    """Serialize the state dict to a buffer and report its size in MiB.

    This is the size of the saved checkpoint, which is the most concrete
    proxy for "deployment payload" we can measure without writing to disk.
    """
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)


def _make_random_batch(
    *,
    batch_size: int,
    src_len: int,
    tgt_len: int,
    vocab_size: int,
    pad_id: int,
    sos_id: int,
    eos_id: int,
    device: torch.device,
    seed: int,
) -> Dict[str, torch.Tensor]:
    """Build a deterministic random training batch with valid SOS/EOS framing."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Reserve token ids 0..3 for special tokens; sample from the rest.
    low = max(4, pad_id + 1, sos_id + 1, eos_id + 1)
    if low >= vocab_size:
        raise ValueError("vocab_size too small for the special-token reservation")

    src = torch.randint(low, vocab_size, (batch_size, src_len), generator=g, dtype=torch.long)

    tgt_body = torch.randint(low, vocab_size, (batch_size, tgt_len - 2), generator=g, dtype=torch.long)
    sos_col = torch.full((batch_size, 1), sos_id, dtype=torch.long)
    eos_col = torch.full((batch_size, 1), eos_id, dtype=torch.long)
    tgt = torch.cat([sos_col, tgt_body, eos_col], dim=1)

    src_mask = torch.ones_like(src, dtype=torch.float32)
    tgt_mask = torch.ones_like(tgt, dtype=torch.float32)

    return {
        "src_input_ids": src.to(device),
        "tgt_input_ids": tgt.to(device),
        "src_attention_mask": src_mask.to(device),
        "tgt_attention_mask": tgt_mask.to(device),
    }


def _build_model(
    *,
    model_size: str,
    vocab_size: int,
    device: torch.device,
    variant: str,
) -> MobileTranslationModel:
    """Construct a model in the requested numerical variant.

    ``fp32``/``fp16``/``bf16`` cast the freshly-initialised model to the
    requested dtype. ``int8`` runs dynamic quantization via the project's
    quantization utilities (which fall back to a custom eager INT8 wrapper
    when ``torchao`` is unavailable).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    model = create_model(vocab_size=vocab_size, model_size=model_size)

    if variant == "int8":
        # Dynamic quantization expects fp32 weights; quantize first, then move
        # to the requested device.
        from lingolite.quantization_utils import apply_dynamic_quantization

        model = apply_dynamic_quantization(model, dtype=torch.qint8)
        model = model.to(device=device)
        return model

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[variant]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 on CPU is not supported; use bf16 or fp32")
    model = model.to(device=device, dtype=dtype)
    return model


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _bench_train_step(
    *,
    model: MobileTranslationModel,
    device: torch.device,
    batch_size: int,
    src_len: int,
    tgt_len: int,
    vocab_size: int,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    warmup: int,
    iters: int,
    seed: int,
    variant: str = "fp32",
    model_size_mb: Optional[float] = None,
) -> ScenarioResult:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = _make_random_batch(
        batch_size=batch_size,
        src_len=src_len,
        tgt_len=tgt_len,
        vocab_size=vocab_size,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        device=device,
        seed=seed,
    )
    # Tokens-processed denominator: target tokens that actually contribute to loss
    # (everything except the trailing label position that gets shifted off).
    target_tokens_per_call = batch_size * (tgt_len - 1)

    def step() -> int:
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(
            src_input_ids=batch["src_input_ids"],
            tgt_input_ids=batch["tgt_input_ids"],
            src_attention_mask=batch["src_attention_mask"],
            tgt_attention_mask=batch["tgt_attention_mask"],
            label_smoothing=0.1,
        )
        loss.backward()
        optimizer.step()
        return target_tokens_per_call

    with _cuda_memory_scope(device) as peak:
        timings_s, _ = _time_iters(step, device=device, warmup=warmup, iters=iters)

    notes = [
        f"batch={batch_size}, src_len={src_len}, tgt_len={tgt_len}",
    ]
    return _summarize(
        "train_step",
        timings_s,
        tokens_per_call=target_tokens_per_call,
        peak_memory_mb=peak(),
        variant=variant,
        model_size_mb=model_size_mb,
        notes=notes,
    )


def _bench_decode(
    *,
    name: str,
    model: MobileTranslationModel,
    device: torch.device,
    batch_size: int,
    src_len: int,
    max_length: int,
    vocab_size: int,
    sos_id: int,
    eos_id: int,
    pad_id: int,
    warmup: int,
    iters: int,
    seed: int,
    num_beams: int,
    variant: str = "fp32",
    model_size_mb: Optional[float] = None,
) -> ScenarioResult:
    model.eval()
    batch = _make_random_batch(
        batch_size=batch_size,
        src_len=src_len,
        tgt_len=4,  # unused for decode benches
        vocab_size=vocab_size,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        device=device,
        seed=seed,
    )
    src = batch["src_input_ids"]
    src_mask = batch["src_attention_mask"]

    if name == "greedy":
        def step() -> int:
            out = model.generate_with_cache(
                src_input_ids=src,
                src_attention_mask=src_mask,
                max_length=max_length,
                sos_token_id=sos_id,
                eos_token_id=eos_id,
            )
            # Total decoded tokens (excluding the SOS we seeded with).
            return int(out.shape[0]) * max(0, int(out.shape[1]) - 1)
    elif name == "beam":
        def step() -> int:
            out = model.generate_beam(
                src_input_ids=src,
                src_attention_mask=src_mask,
                max_length=max_length,
                num_beams=num_beams,
                sos_token_id=sos_id,
                eos_token_id=eos_id,
            )
            return int(out.shape[0]) * max(0, int(out.shape[1]) - 1)
    else:
        raise ValueError(f"Unknown decode scenario: {name}")

    with _cuda_memory_scope(device) as peak:
        timings_s, _ = _time_iters(step, device=device, warmup=warmup, iters=iters)

    notes = [f"batch={batch_size}, src_len={src_len}, max_length={max_length}"]
    if name == "beam":
        notes.append(f"num_beams={num_beams}")

    return _summarize(
        name,
        timings_s,
        tokens_per_call=None,  # decode token count varies with EOS; report latency only
        peak_memory_mb=peak(),
        variant=variant,
        model_size_mb=model_size_mb,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# CLI / formatting
# ---------------------------------------------------------------------------

def _format_table(results: List[ScenarioResult]) -> str:
    headers = [
        "variant",
        "scenario",
        "iters",
        "mean ms",
        "p50 ms",
        "p95 ms",
        "tok/s",
        "peak MB",
        "model MB",
        "notes",
    ]
    rows: List[List[str]] = []
    for r in results:
        rows.append([
            r.variant,
            r.name,
            str(r.iters),
            f"{r.mean_ms:.2f}",
            f"{r.p50_ms:.2f}",
            f"{r.p95_ms:.2f}",
            "-" if r.tokens_per_sec is None else f"{r.tokens_per_sec:.1f}",
            "-" if r.peak_memory_mb is None else f"{r.peak_memory_mb:.1f}",
            "-" if r.model_size_mb is None else f"{r.model_size_mb:.1f}",
            "; ".join(r.notes),
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "  "

    def fmt_row(values: List[str]) -> str:
        return sep.join(v.ljust(widths[i]) for i, v in enumerate(values))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LingoLite performance benchmark")
    p.add_argument("--model-size", choices=["tiny", "small", "medium", "large"], default="small")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=None,
        help="Run the matrix across these model variants. If omitted, runs a single variant per --dtype/--int8.",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Single-variant dtype (ignored when --variants is given).",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        help="Single-variant shortcut for int8 dynamic quantization (ignored when --variants is given).",
    )
    p.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=list(SCENARIOS),
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--src-len", type=int, default=32)
    p.add_argument("--tgt-len", type=int, default=32, help="Target seq length for train_step")
    p.add_argument("--max-length", type=int, default=32, help="Decode budget for greedy/beam")
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    return p


def _resolve_variants(args: argparse.Namespace) -> List[str]:
    """Pick which numerical variants to run.

    Explicit ``--variants`` wins. Otherwise we synthesize a single-variant
    list from ``--dtype`` / ``--int8`` so the legacy CLI keeps working.
    """
    if args.variants:
        return list(args.variants)
    if args.int8:
        return ["int8"]
    return [{"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[args.dtype]]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    variants = _resolve_variants(args)

    pad_id, sos_id, eos_id = 0, 1, 2
    results: List[ScenarioResult] = []
    param_count: Optional[int] = None
    variant_sizes: Dict[str, float] = {}

    for variant in variants:
        try:
            model = _build_model(
                model_size=args.model_size,
                vocab_size=args.vocab_size,
                device=device,
                variant=variant,
            )
        except ValueError as exc:
            # Surface unsupported combinations (e.g. fp16 on CPU) but keep
            # going so the rest of the matrix still reports.
            print(f"[skip] variant={variant}: {exc}", file=sys.stderr)
            continue

        if param_count is None:
            param_count = sum(p.numel() for p in model.parameters())

        size_mb = _measure_model_size_mb(model)
        variant_sizes[variant] = size_mb

        for scenario in args.scenarios:
            if scenario == "train_step" and variant in INFERENCE_ONLY_VARIANTS:
                # int8 dynamic quantization replaces nn.Linear with custom
                # wrappers that don't carry gradients - skip silently.
                continue
            if scenario == "train_step":
                results.append(
                    _bench_train_step(
                        model=model,
                        device=device,
                        batch_size=args.batch_size,
                        src_len=args.src_len,
                        tgt_len=args.tgt_len,
                        vocab_size=args.vocab_size,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        variant=variant,
                        model_size_mb=size_mb,
                    )
                )
            else:
                results.append(
                    _bench_decode(
                        name=scenario,
                        model=model,
                        device=device,
                        batch_size=args.batch_size,
                        src_len=args.src_len,
                        max_length=args.max_length,
                        vocab_size=args.vocab_size,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        num_beams=args.num_beams,
                        variant=variant,
                        model_size_mb=size_mb,
                    )
                )

    if not results:
        print("No variants produced a result; check --variants and device support", file=sys.stderr)
        return 2

    header_info: Dict[str, object] = {
        "model_size": args.model_size,
        "vocab_size": args.vocab_size,
        "param_count": param_count,
        "device": str(device),
        "variants": variants,
        "variant_size_mb": variant_sizes,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }

    if args.json:
        print(json.dumps({"config": header_info, "results": [asdict(r) for r in results]}, indent=2))
    else:
        param_label = f"{param_count:,}" if param_count is not None else "?"
        print(
            f"Model: {args.model_size} ({param_label} params)  device={device}  "
            f"variants={','.join(variants)}"
        )
        if header_info["cuda_device"]:
            print(f"CUDA device: {header_info['cuda_device']}")
        print()
        print(_format_table(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
