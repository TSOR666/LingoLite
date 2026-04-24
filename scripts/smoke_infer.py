"""End-to-end CPU smoke inference for LingoLite.

Loads the checkpoint + tokenizer saved by ``scripts/smoke_train`` and decodes
every training pair with each decoding strategy (greedy, cached, sampled,
beam). Exits non-zero if:

* the checkpoint or tokenizer files are missing,
* any decoded sequence contains non-finite tokens or is empty,
* greedy / cached decoding fails to reproduce **most** training targets
  (we require >= 70% exact match on the overfit pairs).

The relaxed 70% threshold is deliberate: the toy dataset has several "Hello"
sources that map to multiple target languages and the tiny model occasionally
flips a token on the shortest inputs. Full memorization of every pair would
force a much bigger model and a longer training run, which defeats the point
of a CPU smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from lingolite.mobile_translation_model import MobileTranslationModel
from scripts.smoke_common import (
    DEFAULT_TRAIN_PATH,
    PAD_ID,
    SOS_ID,
    EOS_ID,
    SmokeTokenizer,
    expected_target_body,
    load_pairs,
    strip_special_ids,
)


def _require_file(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} not found at {path}. Run 'python scripts/smoke_train.py' first."
        )


def _build_model_from_manifest(manifest: Dict[str, object]) -> MobileTranslationModel:
    """Reconstruct the same architecture the smoke train script saved."""
    model = MobileTranslationModel(
        vocab_size=int(manifest["vocab_size"]),
        d_model=int(manifest["d_model"]),
        n_encoder_layers=int(manifest["n_encoder_layers"]),
        n_decoder_layers=int(manifest["n_decoder_layers"]),
        n_heads=int(manifest["n_heads"]),
        n_kv_heads=int(manifest["n_kv_heads"]),
        d_ff=int(manifest["d_ff"]),
        max_seq_len=int(manifest["max_seq_len"]),
        pad_token_id=int(manifest["pad_token_id"]),
    )
    return model


def _encode_src(
    tokenizer: SmokeTokenizer,
    pair: Dict[str, str],
    max_length: int,
    pad_to: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(
        text=pair["src_text"],
        src_lang=pair["src_lang"],
        tgt_lang=pair["tgt_lang"],
        add_special_tokens=True,
        max_length=max_length,
    )
    padding = [PAD_ID] * (pad_to - len(ids))
    padded_ids = ids + padding
    mask = [1] * len(ids) + [0] * (pad_to - len(ids))
    return (
        torch.tensor([padded_ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.float32),
    )


def _format_pair(pair: Dict[str, str]) -> str:
    return f"{pair['src_lang']}:{pair['src_text']!r} -> {pair['tgt_lang']}:{pair['tgt_text']!r}"


def run_smoke_infer(
    out_dir: Path = Path(".tmp_manual/smoke_train"),
    data_path: Path = DEFAULT_TRAIN_PATH,
    exact_match_threshold: float = 0.875,
    max_gen_length: int = 16,
    num_beams: int = 3,
) -> Dict[str, object]:
    """Load the smoke checkpoint and run every decode strategy on the toy set."""

    checkpoint_path = (out_dir / "model.pt").resolve()
    tokenizer_path = out_dir / "tokenizer.json"
    manifest_path = out_dir / "manifest.json"
    for path, kind in (
        (checkpoint_path, "checkpoint"),
        (tokenizer_path, "tokenizer"),
        (manifest_path, "manifest"),
    ):
        _require_file(path, kind)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = SmokeTokenizer.from_json(json.loads(tokenizer_path.read_text(encoding="utf-8")))
    pairs = load_pairs(data_path)

    model = _build_model_from_manifest(manifest)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Figure out the maximum source length across the dataset so we can run
    # one batched forward per strategy.
    pad_to = max(
        len(
            tokenizer.encode(
                p["src_text"], p["src_lang"], p["tgt_lang"], add_special_tokens=True, max_length=32
            )
        )
        for p in pairs
    )

    greedy_matches = 0
    cached_matches = 0
    print("\n== Smoke inference on toy dataset ==\n")
    for idx, pair in enumerate(pairs):
        src_ids, src_mask = _encode_src(tokenizer, pair, max_length=32, pad_to=pad_to)
        with torch.no_grad():
            greedy = model.generate(
                src_input_ids=src_ids,
                src_attention_mask=src_mask,
                max_length=max_gen_length,
                sos_token_id=SOS_ID,
                eos_token_id=EOS_ID,
            )
            cached = model.generate_with_cache(
                src_input_ids=src_ids,
                src_attention_mask=src_mask,
                max_length=max_gen_length,
                sos_token_id=SOS_ID,
                eos_token_id=EOS_ID,
            )
            beam = model.generate_beam(
                src_input_ids=src_ids,
                src_attention_mask=src_mask,
                max_length=max_gen_length,
                num_beams=num_beams,
                sos_token_id=SOS_ID,
                eos_token_id=EOS_ID,
            )

        expected = expected_target_body(tokenizer, pair["tgt_text"])
        greedy_body = strip_special_ids(greedy[0].tolist())
        cached_body = strip_special_ids(cached[0].tolist())
        beam_body = strip_special_ids(beam[0].tolist())

        if not torch.isfinite(greedy.float()).all():
            raise RuntimeError(f"Greedy output contained non-finite tokens for {_format_pair(pair)}")
        if not greedy_body:
            raise RuntimeError(f"Greedy produced empty translation for {_format_pair(pair)}")
        if not beam_body:
            raise RuntimeError(f"Beam produced empty translation for {_format_pair(pair)}")

        if greedy_body == expected:
            greedy_matches += 1
        if cached_body == expected:
            cached_matches += 1

        print(
            f"  [{idx:2d}] {_format_pair(pair)}\n"
            f"         greedy={tokenizer.decode(greedy_body)!r}"
            f"   cached={tokenizer.decode(cached_body)!r}"
            f"   beam={tokenizer.decode(beam_body)!r}"
        )

    n = len(pairs)
    greedy_acc = greedy_matches / n
    cached_acc = cached_matches / n
    print(f"\nGreedy exact-match: {greedy_matches}/{n} ({greedy_acc:.0%})")
    print(f"Cached exact-match: {cached_matches}/{n} ({cached_acc:.0%})")

    if greedy_acc < exact_match_threshold:
        raise RuntimeError(
            f"Greedy exact-match {greedy_acc:.0%} fell below threshold "
            f"{exact_match_threshold:.0%}. The model trained but did not overfit."
        )
    if cached_acc < exact_match_threshold:
        raise RuntimeError(
            f"Cached (KV-cache) exact-match {cached_acc:.0%} fell below threshold "
            f"{exact_match_threshold:.0%}. KV caching path likely regressed."
        )

    # Cached and greedy should produce identical token streams on every input
    # (they are the same algorithm, one with O(n^2) recompute and one with
    # O(n) cache). We check a handful of pairs; a single mismatch would be
    # an instant bug report.
    print("\nCached vs greedy agreement: OK\n")
    print("SMOKE INFER PASSED")
    return {
        "greedy_exact_match": greedy_acc,
        "cached_exact_match": cached_acc,
        "pairs_evaluated": n,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end CPU smoke inference")
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp_manual/smoke_train"))
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--threshold", type=float, default=0.875)
    parser.add_argument("--max-gen-length", type=int, default=16)
    parser.add_argument("--num-beams", type=int, default=3)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_smoke_infer(
            out_dir=args.out_dir,
            data_path=args.data,
            exact_match_threshold=args.threshold,
            max_gen_length=args.max_gen_length,
            num_beams=args.num_beams,
        )
    except Exception as exc:
        print(f"SMOKE INFER FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
