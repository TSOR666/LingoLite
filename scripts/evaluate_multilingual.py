"""Multilingual translation quality evaluation with length-bucket reporting.

Loads a translation checkpoint + tokenizer and a JSON dataset shaped like
``examples/data/tiny_dataset.json`` (a list of dicts with ``src_text``,
``tgt_text``, ``src_lang``, ``tgt_lang``) and reports per-language-pair
BLEU / chrF / chrF++ plus length-bucketed BLEU within each pair.

Why this exists: ``scripts/evaluate_model.py`` measures a single (src, tgt)
pair from parallel text files. For multilingual models we want to see how
quality varies across pairs *and* across input length without re-running
the script per pair. A regression on long sentences typically shows up as
a length-bucket cliff long before the aggregate BLEU moves.

Example::

    python scripts/evaluate_multilingual.py \\
        --checkpoint checkpoints/best_model.pt \\
        --tokenizer checkpoints/tokenizer \\
        --dataset examples/data/tiny_dataset_val.json \\
        --max-length 64 --num-beams 4 --output eval_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import torch

try:
    import sacrebleu  # type: ignore[import-untyped]
except ImportError:
    print("ERROR: sacrebleu not installed. Install with: pip install sacrebleu", file=sys.stderr)
    raise

from lingolite.mobile_translation_model import MobileTranslationModel, load_model_from_checkpoint
from lingolite.translation_tokenizer import TranslationTokenizer


# Default source-token length buckets. ``inf`` is represented as ``None`` to
# stay JSON-serialisable.
_DEFAULT_BUCKETS: Tuple[Tuple[int, Optional[int]], ...] = (
    (1, 10),
    (11, 20),
    (21, 50),
    (51, None),
)


@dataclass
class PairResult:
    """Quality metrics for a single (src_lang, tgt_lang) pair."""

    src_lang: str
    tgt_lang: str
    n: int
    bleu: float
    chrf: float
    chrf_pp: float
    buckets: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class EvalReport:
    """Top-level evaluation report for a checkpoint."""

    checkpoint: str
    dataset: str
    num_pairs_evaluated: int
    num_examples: int
    overall_bleu: float
    overall_chrf: float
    overall_chrf_pp: float
    per_pair: List[PairResult]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> List[Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty JSON list of translation pairs")
    required = {"src_text", "tgt_text", "src_lang", "tgt_lang"}
    for i, item in enumerate(payload):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"dataset item {i} missing fields: {sorted(missing)}")
    return payload


def _group_by_pair(
    examples: Sequence[Dict[str, str]],
) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for ex in examples:
        key = (ex["src_lang"], ex["tgt_lang"])
        grouped.setdefault(key, []).append(ex)
    return grouped


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path, tokenizer: TranslationTokenizer, device: torch.device) -> MobileTranslationModel:
    """Reconstruct the model from a checkpoint, falling back to a sensible default."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = load_model_from_checkpoint(
        checkpoint,
        fallback_vocab_size=tokenizer.get_vocab_size(),
        fallback_model_size="small",
    )
    model.to(device)
    model.eval()
    return model


def _translate_batch(
    *,
    model: MobileTranslationModel,
    tokenizer: TranslationTokenizer,
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_beams: int,
    device: torch.device,
) -> List[str]:
    """Tokenize a batch, run generation, return decoded translations."""
    encoded = tokenizer.batch_encode(
        texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        padding=True,
        max_length=max_length,
        return_tensors=True,
    )
    src_ids = cast(torch.Tensor, encoded["input_ids"]).to(device)
    src_mask = cast(torch.Tensor, encoded["attention_mask"]).to(device)

    with torch.inference_mode():
        if num_beams > 1:
            generated = model.generate_beam(
                src_input_ids=src_ids,
                src_attention_mask=src_mask,
                max_length=max_length,
                num_beams=num_beams,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            generated = model.generate_with_cache(
                src_input_ids=src_ids,
                src_attention_mask=src_mask,
                max_length=max_length,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    return tokenizer.batch_decode(generated.tolist())


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _corpus_bleu(hyps: Sequence[str], refs: Sequence[str], lowercase: bool) -> float:
    if not hyps:
        return 0.0
    return float(sacrebleu.corpus_bleu(list(hyps), [list(refs)], lowercase=lowercase).score)


def _corpus_chrf(hyps: Sequence[str], refs: Sequence[str], word_order: int = 0) -> float:
    if not hyps:
        return 0.0
    return float(sacrebleu.corpus_chrf(list(hyps), [list(refs)], word_order=word_order).score)


def _bucket_label(low: int, high: Optional[int]) -> str:
    return f"{low}-{high}" if high is not None else f"{low}+"


def _bucket_examples(
    indices: Sequence[int],
    src_token_lengths: Sequence[int],
    buckets: Sequence[Tuple[int, Optional[int]]],
) -> Dict[Tuple[int, Optional[int]], List[int]]:
    """Group example indices into source-length buckets."""
    out: Dict[Tuple[int, Optional[int]], List[int]] = {b: [] for b in buckets}
    for idx in indices:
        length = src_token_lengths[idx]
        for low, high in buckets:
            if length >= low and (high is None or length <= high):
                out[(low, high)].append(idx)
                break
    return out


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    dataset_path: Path,
    max_length: int = 128,
    batch_size: int = 16,
    num_beams: int = 1,
    buckets: Sequence[Tuple[int, Optional[int]]] = _DEFAULT_BUCKETS,
    lowercase: bool = False,
    device: Optional[str] = None,
    max_pairs: Optional[int] = None,
    max_examples_per_pair: Optional[int] = None,
) -> EvalReport:
    """Run translation + scoring across every language pair in ``dataset_path``."""

    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = TranslationTokenizer.from_pretrained(tokenizer_path)
    model = _load_model(checkpoint, tokenizer, device_obj)

    examples = _load_dataset(dataset_path)
    grouped = _group_by_pair(examples)
    pair_keys = list(grouped.keys())
    if max_pairs is not None:
        pair_keys = pair_keys[:max_pairs]

    all_hyps: List[str] = []
    all_refs: List[str] = []
    per_pair: List[PairResult] = []

    for src_lang, tgt_lang in pair_keys:
        pair_examples = grouped[(src_lang, tgt_lang)]
        if max_examples_per_pair is not None:
            pair_examples = pair_examples[:max_examples_per_pair]

        sources = [ex["src_text"] for ex in pair_examples]
        refs = [ex["tgt_text"] for ex in pair_examples]

        # Pre-compute source token lengths for length bucketing. We use the
        # tokenizer's plain encode (no special tokens) as the proxy for "input
        # complexity" since BLEU is computed on the target side.
        src_lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sources]

        # Translate in batches.
        hyps: List[str] = []
        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            hyps.extend(
                _translate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    texts=batch,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_length=max_length,
                    num_beams=num_beams,
                    device=device_obj,
                )
            )

        # Whole-pair scores.
        bleu = _corpus_bleu(hyps, refs, lowercase)
        chrf = _corpus_chrf(hyps, refs, word_order=0)
        chrf_pp = _corpus_chrf(hyps, refs, word_order=2)

        # Length-bucket scores.
        bucket_groups = _bucket_examples(range(len(sources)), src_lengths, buckets)
        bucket_results: List[Dict[str, object]] = []
        for low, high in buckets:
            idxs = bucket_groups[(low, high)]
            if not idxs:
                bucket_results.append(
                    {"bucket": _bucket_label(low, high), "n": 0, "bleu": None, "chrf": None}
                )
                continue
            b_hyps = [hyps[i] for i in idxs]
            b_refs = [refs[i] for i in idxs]
            bucket_results.append(
                {
                    "bucket": _bucket_label(low, high),
                    "n": len(idxs),
                    "bleu": _corpus_bleu(b_hyps, b_refs, lowercase),
                    "chrf": _corpus_chrf(b_hyps, b_refs, word_order=0),
                }
            )

        per_pair.append(
            PairResult(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                n=len(sources),
                bleu=bleu,
                chrf=chrf,
                chrf_pp=chrf_pp,
                buckets=bucket_results,
            )
        )

        all_hyps.extend(hyps)
        all_refs.extend(refs)

    return EvalReport(
        checkpoint=str(checkpoint),
        dataset=str(dataset_path),
        num_pairs_evaluated=len(per_pair),
        num_examples=len(all_hyps),
        overall_bleu=_corpus_bleu(all_hyps, all_refs, lowercase),
        overall_chrf=_corpus_chrf(all_hyps, all_refs, word_order=0),
        overall_chrf_pp=_corpus_chrf(all_hyps, all_refs, word_order=2),
        per_pair=per_pair,
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _format_report(report: EvalReport) -> str:
    lines: List[str] = []
    lines.append(f"Checkpoint: {report.checkpoint}")
    lines.append(f"Dataset:    {report.dataset}")
    lines.append(
        f"Examples: {report.num_examples}  Pairs: {report.num_pairs_evaluated}"
    )
    lines.append("")
    lines.append(
        f"OVERALL  BLEU={report.overall_bleu:5.2f}  "
        f"chrF={report.overall_chrf:5.2f}  chrF++={report.overall_chrf_pp:5.2f}"
    )
    lines.append("")
    lines.append("Per-pair:")
    lines.append(f"  {'pair':<8} {'n':>5}  {'BLEU':>6}  {'chrF':>6}  {'chrF++':>6}")
    lines.append(f"  {'-'*8} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}")
    for pair in report.per_pair:
        label = f"{pair.src_lang}->{pair.tgt_lang}"
        lines.append(
            f"  {label:<8} {pair.n:>5}  {pair.bleu:>6.2f}  {pair.chrf:>6.2f}  {pair.chrf_pp:>6.2f}"
        )

    lines.append("")
    lines.append("Length buckets (source tokens):")
    for pair in report.per_pair:
        label = f"{pair.src_lang}->{pair.tgt_lang}"
        lines.append(f"  [{label}]")
        for b in pair.buckets:
            n = cast(int, b["n"])
            if n == 0:
                lines.append(f"    {cast(str, b['bucket']):<8} n=0   (no examples)")
                continue
            lines.append(
                f"    {cast(str, b['bucket']):<8} n={n:<4}  "
                f"BLEU={cast(float, b['bleu']):5.2f}  chrF={cast(float, b['chrf']):5.2f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_buckets(raw: Optional[str]) -> Tuple[Tuple[int, Optional[int]], ...]:
    if not raw:
        return _DEFAULT_BUCKETS
    out: List[Tuple[int, Optional[int]]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.endswith("+"):
            low = int(chunk[:-1])
            out.append((low, None))
        elif "-" in chunk:
            low_s, high_s = chunk.split("-", 1)
            out.append((int(low_s), int(high_s)))
        else:
            raise ValueError(f"unrecognised bucket spec {chunk!r} (use '1-10' or '50+')")
    if not out:
        raise ValueError("--buckets parsed to an empty list")
    return tuple(out)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint (.pt)")
    p.add_argument("--tokenizer", type=Path, required=True, help="Tokenizer directory")
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSON list of {src_text, tgt_text, src_lang, tgt_lang} examples",
    )
    p.add_argument("--output", type=Path, default=None, help="Optional path to write JSON report")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=1, help=">1 enables beam search")
    p.add_argument("--lowercase", action="store_true", help="Lowercase before scoring")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument(
        "--buckets",
        type=str,
        default=None,
        help='Comma-separated source-token-length buckets, e.g. "1-10,11-20,21-50,51+"',
    )
    p.add_argument("--max-pairs", type=int, default=None, help="Cap distinct language pairs")
    p.add_argument(
        "--max-examples-per-pair",
        type=int,
        default=None,
        help="Cap examples per pair (useful for smoke tests)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    buckets = _parse_buckets(args.buckets)

    report = evaluate(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        dataset_path=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        buckets=buckets,
        lowercase=args.lowercase,
        device=args.device,
        max_pairs=args.max_pairs,
        max_examples_per_pair=args.max_examples_per_pair,
    )

    print(_format_report(report))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Use ``asdict`` so dataclasses serialise cleanly; convert PairResult.
        payload = {
            **asdict(report),
            "per_pair": [asdict(p) for p in report.per_pair],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
