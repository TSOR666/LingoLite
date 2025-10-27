"""BLEU evaluation utilities using sacrebleu.

This script computes corpus-level BLEU scores between reference and
hypothesis translations. It supports plain text files (one sentence per
line) and JSON/JSONL files containing a specified field for references
and hypotheses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import sacrebleu


def _load_from_text(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _load_from_json(path: Path, field: str) -> List[str]:
    data: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                data.append(str(obj[field]))
        else:
            payload = json.load(f)
            if isinstance(payload, dict):
                payload = payload[field]
            for item in payload:
                if isinstance(item, dict):
                    data.append(str(item[field]))
                else:
                    data.append(str(item))
    return data


def load_sentences(path: Path, field: str | None = None) -> List[str]:
    """Load sentences from text or JSON files."""

    if field is None and path.suffix in {".json", ".jsonl"}:
        raise ValueError("JSON inputs require --field to specify the target key")

    if path.suffix in {".json", ".jsonl"}:
        assert field is not None
        return _load_from_json(path, field)

    return _load_from_text(path)


def compute_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    tokenizer: str = "13a",
    lowercase: bool = False,
    smooth_method: str = "exp",
) -> sacrebleu.metrics.bleu.BLEUScore:
    """Compute corpus BLEU using sacrebleu."""

    return sacrebleu.corpus_bleu(
        sys_stream=hypotheses,
        ref_streams=list(references),
        smooth_method=smooth_method,
        lowercase=lowercase,
        tokenize=tokenizer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BLEU scores with sacrebleu")
    parser.add_argument("hypotheses", type=Path, help="File containing model hypotheses")
    parser.add_argument("references", nargs="+", type=Path, help="Reference translation files")
    parser.add_argument(
        "--hyp-field",
        type=str,
        default=None,
        help="JSON key for hypothesis sentences (required for JSON/JSONL)",
    )
    parser.add_argument(
        "--ref-field",
        type=str,
        default=None,
        help="JSON key for reference sentences (required for JSON/JSONL)",
    )
    parser.add_argument("--tokenizer", type=str, default="13a", help="sacrebleu tokenizer")
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase hypotheses and references before scoring",
    )
    parser.add_argument(
        "--smooth",
        type=str,
        default="exp",
        help="Smoothing method (defaults to exp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hyp_sentences = load_sentences(args.hypotheses, args.hyp_field)
    ref_streams = [load_sentences(path, args.ref_field) for path in args.references]

    if any(len(stream) != len(hyp_sentences) for stream in ref_streams):
        raise ValueError("Reference and hypothesis files must contain the same number of sentences")

    bleu = compute_bleu(
        hypotheses=hyp_sentences,
        references=ref_streams,
        tokenizer=args.tokenizer,
        lowercase=args.lowercase,
        smooth_method=args.smooth,
    )

    print(f"BLEU = {bleu.score:.2f}")
    print(f"Precisions: {bleu.precisions}")
    print(f"BP: {bleu.bp:.4f}, ratio: {bleu.sys_len}/{bleu.ref_len}")


if __name__ == "__main__":
    main()
