"""Tests for scripts/evaluate_multilingual.py.

Two layers:

* Unit-level tests for the pure-Python helpers (bucket parsing, grouping by
  pair, length bucketing) - cheap and run for every commit.
* An end-to-end smoke test that trains a real SentencePiece tokenizer on
  the tiny dataset, saves an untrained tiny-model checkpoint, and runs the
  full ``evaluate()`` pipeline. We don't assert on the BLEU value (the
  model is random) - we assert on report *shape* and bucket invariants,
  which is what regressions tend to break.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator, Tuple

import pytest
import sentencepiece as spm  # type: ignore[import-untyped]
import torch

from lingolite.mobile_translation_model import MobileTranslationModel
from lingolite.translation_tokenizer import TranslationTokenizer
from scripts.evaluate_multilingual import (
    EvalReport,
    PairResult,
    _bucket_examples,
    _bucket_label,
    _format_report,
    _group_by_pair,
    _parse_buckets,
    evaluate,
)
from tests.tmp_utils import writable_tmp_dir


_TINY_DATASET = Path("examples/data/tiny_dataset.json")


# ---------------------------------------------------------------------------
# Pure-Python helper tests
# ---------------------------------------------------------------------------

class TestBucketParsing:
    def test_default_buckets(self) -> None:
        assert _parse_buckets(None) == ((1, 10), (11, 20), (21, 50), (51, None))

    def test_custom_buckets(self) -> None:
        assert _parse_buckets("1-5,6-20,21+") == ((1, 5), (6, 20), (21, None))

    def test_open_ended_top_bucket(self) -> None:
        assert _parse_buckets("100+") == ((100, None),)

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty list"):
            _parse_buckets(",,")

    def test_unrecognised_token_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognised"):
            _parse_buckets("hello")

    def test_label_formatting(self) -> None:
        assert _bucket_label(1, 10) == "1-10"
        assert _bucket_label(50, None) == "50+"


class TestGroupingAndBucketing:
    def test_group_by_pair(self) -> None:
        examples = [
            {"src_text": "a", "tgt_text": "A", "src_lang": "en", "tgt_lang": "es"},
            {"src_text": "b", "tgt_text": "B", "src_lang": "en", "tgt_lang": "fr"},
            {"src_text": "c", "tgt_text": "C", "src_lang": "en", "tgt_lang": "es"},
            {"src_text": "d", "tgt_text": "D", "src_lang": "es", "tgt_lang": "en"},
        ]
        groups = _group_by_pair(examples)
        assert len(groups[("en", "es")]) == 2
        assert len(groups[("en", "fr")]) == 1
        assert len(groups[("es", "en")]) == 1

    def test_bucket_examples_assigns_each_index_once(self) -> None:
        lengths = [3, 15, 30, 100, 7]
        buckets = ((1, 10), (11, 20), (21, 50), (51, None))
        result = _bucket_examples(range(len(lengths)), lengths, buckets)
        assert result[(1, 10)] == [0, 4]
        assert result[(11, 20)] == [1]
        assert result[(21, 50)] == [2]
        assert result[(51, None)] == [3]
        # Every index got bucketed exactly once.
        flat = sum(result.values(), [])
        assert sorted(flat) == list(range(len(lengths)))

    def test_bucket_examples_excludes_below_minimum(self) -> None:
        # Length 0 falls below the smallest bucket (1-10) and should be dropped.
        result = _bucket_examples(range(2), [0, 5], ((1, 10),))
        assert result[(1, 10)] == [1]


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_artifacts() -> Iterator[Tuple[Path, Path, Path]]:
    """Train an SP tokenizer + save an untrained tiny-model checkpoint once.

    Returns (checkpoint_path, tokenizer_dir, dataset_path). Uses a project-local
    ``.tmp_pytest`` root to dodge Windows ``%TEMP%`` permission issues.
    """
    if not _TINY_DATASET.exists():
        pytest.skip(f"missing tiny dataset at {_TINY_DATASET}")

    with writable_tmp_dir("evaluate_multilingual_") as out_dir:
        pairs = json.loads(_TINY_DATASET.read_text(encoding="utf-8"))
        corpus_path = out_dir / "corpus.txt"
        with corpus_path.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(p["src_text"].strip() + "\n")
                fh.write(p["tgt_text"].strip() + "\n")

        languages = ["en", "es", "fr", "de", "it", "da"]
        sp_prefix = out_dir / "tok"
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(sp_prefix),
            vocab_size=100,
            character_coverage=1.0,
            model_type="unigram",
            pad_id=0,
            unk_id=3,
            bos_id=1,
            eos_id=2,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            user_defined_symbols=[f"<{lang}>" for lang in languages] + ["<src>", "<tgt>"],
            num_threads=1,
        )

        tokenizer = TranslationTokenizer(languages=languages, vocab_size=100, model_prefix="tok")
        tokenizer.load(str(sp_prefix.with_suffix(".model")))

        tokenizer_dir = out_dir / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "languages": languages,
                    "vocab_size": 100,
                    "special_tokens": tokenizer.special_tokens,
                    "model_prefix": "tok",
                }
            ),
            encoding="utf-8",
        )
        shutil.copyfile(sp_prefix.with_suffix(".model"), tokenizer_dir / "tok.model")

        config = dict(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=64,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=2,
            n_kv_heads=1,
            d_ff=128,
            max_seq_len=64,
        )
        model = MobileTranslationModel(**config)
        ckpt_path = out_dir / "model.pt"
        torch.save({"config": config, "model_state_dict": model.state_dict()}, ckpt_path)

        # A small slice of the tiny dataset that spans multiple pairs.
        small = [p for p in pairs if (p["src_lang"], p["tgt_lang"]) in {("en", "es"), ("en", "fr"), ("es", "en")}]
        ds_path = out_dir / "subset.json"
        ds_path.write_text(json.dumps(small[:9]), encoding="utf-8")

        yield ckpt_path, tokenizer_dir, ds_path


class TestEndToEnd:
    def test_report_shape(self, trained_artifacts: Tuple[Path, Path, Path]) -> None:
        ckpt, tok_dir, ds_path = trained_artifacts
        report = evaluate(
            checkpoint=ckpt,
            tokenizer_path=tok_dir,
            dataset_path=ds_path,
            max_length=16,
            batch_size=2,
            num_beams=1,
            device="cpu",
        )
        assert isinstance(report, EvalReport)
        assert report.num_examples > 0
        assert report.num_pairs_evaluated >= 1
        # Untrained model can't produce real translations, so we don't assert
        # on the score itself - we assert it's a finite number in [0, 100].
        assert 0.0 <= report.overall_bleu <= 100.0
        assert 0.0 <= report.overall_chrf <= 100.0
        for pair in report.per_pair:
            assert isinstance(pair, PairResult)
            assert pair.n > 0
            assert sum(int(b["n"]) for b in pair.buckets) == pair.n

    def test_format_report_returns_string(self, trained_artifacts: Tuple[Path, Path, Path]) -> None:
        ckpt, tok_dir, ds_path = trained_artifacts
        report = evaluate(
            checkpoint=ckpt,
            tokenizer_path=tok_dir,
            dataset_path=ds_path,
            max_length=16,
            batch_size=2,
            num_beams=1,
            device="cpu",
            max_examples_per_pair=3,
        )
        rendered = _format_report(report)
        assert "OVERALL" in rendered
        assert "Per-pair:" in rendered
        assert "Length buckets" in rendered

    def test_max_examples_per_pair_caps_size(self, trained_artifacts: Tuple[Path, Path, Path]) -> None:
        ckpt, tok_dir, ds_path = trained_artifacts
        report = evaluate(
            checkpoint=ckpt,
            tokenizer_path=tok_dir,
            dataset_path=ds_path,
            max_length=16,
            batch_size=2,
            num_beams=1,
            device="cpu",
            max_examples_per_pair=2,
        )
        for pair in report.per_pair:
            assert pair.n <= 2

    def test_beam_search_path_runs(self, trained_artifacts: Tuple[Path, Path, Path]) -> None:
        ckpt, tok_dir, ds_path = trained_artifacts
        report = evaluate(
            checkpoint=ckpt,
            tokenizer_path=tok_dir,
            dataset_path=ds_path,
            max_length=12,
            batch_size=2,
            num_beams=2,
            device="cpu",
            max_examples_per_pair=2,
        )
        assert report.num_examples > 0
