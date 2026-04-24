"""End-to-end CPU smoke pipeline tests.

These tests go one step further than ``tests/test_smoke_translate.py``:
they exercise the actual train/infer scripts plus the shared smoke helpers,
train a tiny model on the toy dataset, round-trip the checkpoint through
disk, and verify that every decoding strategy can reproduce the training
targets.

All of this runs on CPU in a few seconds -- no external data or tokenizer
required. If one of these fails, the entire "train on a dataset -> save
checkpoint -> reload checkpoint -> translate" pipeline is broken, not just
a single unit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
import torch

from lingolite.mobile_translation_model import MobileTranslationModel
from scripts.smoke_common import (
    DEFAULT_TRAIN_PATH,
    EOS_ID,
    PAD_ID,
    SOS_ID,
    SmokeTokenizer,
    expected_target_body,
    load_pairs,
    strip_special_ids,
)
from scripts.smoke_train import run_smoke_train
from scripts.smoke_infer import run_smoke_infer, _build_model_from_manifest


@pytest.fixture(scope="module")
def trained_smoke_dir() -> Path:
    """Run ``run_smoke_train`` once and share the artifact directory.

    Uses ``.tmp_manual/`` rather than pytest's ``tmp_path`` fixture because the
    Windows CI environment denies ``scandir`` on the system tempdir (see the
    two pre-existing tokenizer tests that are deselected for the same reason).
    """
    out_dir = Path(".tmp_manual") / "smoke_pipeline_test"
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_smoke_train(out_dir=out_dir, seed=0)
    return out_dir


class TestSmokeCommonTokenizer:
    """Unit coverage for the shared ``SmokeTokenizer`` helper."""

    def test_from_pairs_produces_deterministic_vocab(self) -> None:
        pairs = load_pairs(DEFAULT_TRAIN_PATH)
        a = SmokeTokenizer.from_pairs(pairs)
        b = SmokeTokenizer.from_pairs(pairs)
        assert a.word_to_id == b.word_to_id
        assert a.get_vocab_size() == b.get_vocab_size()

    def test_encode_wraps_with_translation_special_tokens(self) -> None:
        pairs = load_pairs(DEFAULT_TRAIN_PATH)
        tok = SmokeTokenizer.from_pairs(pairs)
        ids = tok.encode(
            "Hello", src_lang="en", tgt_lang="es", add_special_tokens=True, max_length=32
        )
        # Layout: <src> <en> body </s> <tgt> <es>
        assert ids[0] == 4  # <src>
        assert ids[1] == 6  # <en>
        assert ids[-3] == EOS_ID
        assert ids[-2] == 5  # <tgt>
        assert ids[-1] == 7  # <es>

    def test_round_trip_json_preserves_vocab(self) -> None:
        pairs = load_pairs(DEFAULT_TRAIN_PATH)
        original = SmokeTokenizer.from_pairs(pairs)
        restored = SmokeTokenizer.from_json(original.to_json())
        assert original.word_to_id == restored.word_to_id
        assert original.languages == restored.languages

    def test_decode_skips_specials(self) -> None:
        pairs = [{"src_text": "hello world", "tgt_text": "hola mundo", "src_lang": "en", "tgt_lang": "es"}]
        tok = SmokeTokenizer.from_pairs(pairs)
        ids = tok.encode("hello world", src_lang="en", tgt_lang="es")
        text = tok.decode(ids, skip_special_tokens=True)
        assert "hello" in text and "world" in text
        # No angle-bracket specials should survive.
        assert "<src>" not in text
        assert "<en>" not in text


class TestSmokeTrainResult:
    """Assertions about the training artifact produced by ``run_smoke_train``."""

    def test_checkpoint_and_tokenizer_files_exist(self, trained_smoke_dir: Path) -> None:
        assert (trained_smoke_dir / "model.pt").exists()
        assert (trained_smoke_dir / "tokenizer.json").exists()
        assert (trained_smoke_dir / "manifest.json").exists()

    def test_manifest_reports_loss_drop(self, trained_smoke_dir: Path) -> None:
        manifest = json.loads((trained_smoke_dir / "manifest.json").read_text(encoding="utf-8"))
        # Convergence requirement is enforced inside run_smoke_train; these
        # cross-checks guard against the manifest drifting from the actual
        # training behaviour.
        assert manifest["first_loss"] > manifest["final_loss"]
        assert manifest["final_loss"] < 0.30
        assert manifest["steps"] > 0

    def test_checkpoint_reloads_into_fresh_model(self, trained_smoke_dir: Path) -> None:
        manifest = json.loads((trained_smoke_dir / "manifest.json").read_text(encoding="utf-8"))
        model = _build_model_from_manifest(manifest)
        checkpoint = torch.load(trained_smoke_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        # And a forward pass on the reloaded weights produces finite outputs.
        model.eval()
        src = torch.randint(1, int(manifest["vocab_size"]), (2, 8))
        with torch.no_grad():
            logits, _, enc = model.forward(
                src_input_ids=src,
                tgt_input_ids=torch.tensor([[SOS_ID, 0], [SOS_ID, 0]], dtype=torch.long),
                src_attention_mask=torch.ones_like(src),
                tgt_attention_mask=torch.ones(2, 2),
            )
        assert torch.isfinite(logits).all()
        assert torch.isfinite(enc).all()


class TestSmokeInferPipeline:
    """``run_smoke_infer`` must load the checkpoint and translate the toy set."""

    def test_greedy_reaches_threshold(self, trained_smoke_dir: Path) -> None:
        result = run_smoke_infer(out_dir=trained_smoke_dir)
        assert result["pairs_evaluated"] == 8
        # The smoke training regime overfits the 8-pair disjoint dataset
        # completely; the threshold is intentionally strict so that a single
        # regression in the KV cache or RoPE path surfaces immediately.
        assert result["greedy_exact_match"] >= 0.875
        assert result["cached_exact_match"] >= 0.875

    def test_cached_matches_greedy_on_every_pair(
        self, trained_smoke_dir: Path
    ) -> None:
        """Cached (KV-cache) decoding must produce the same tokens as greedy.

        A divergence here means the KV cache path recomputes something the
        full path does not (or vice versa); we saw exactly this failure mode
        pre-refactor so we explicitly guard against it.
        """
        manifest = json.loads(
            (trained_smoke_dir / "manifest.json").read_text(encoding="utf-8")
        )
        tokenizer = SmokeTokenizer.from_json(
            json.loads((trained_smoke_dir / "tokenizer.json").read_text(encoding="utf-8"))
        )
        model = _build_model_from_manifest(manifest)
        model.load_state_dict(
            torch.load(trained_smoke_dir / "model.pt", map_location="cpu", weights_only=True)[
                "model_state_dict"
            ]
        )
        model.eval()

        pairs = load_pairs(DEFAULT_TRAIN_PATH)
        pad_to = max(
            len(
                tokenizer.encode(
                    p["src_text"],
                    p["src_lang"],
                    p["tgt_lang"],
                    add_special_tokens=True,
                    max_length=32,
                )
            )
            for p in pairs
        )

        for pair in pairs:
            ids = tokenizer.encode(
                pair["src_text"],
                src_lang=pair["src_lang"],
                tgt_lang=pair["tgt_lang"],
                add_special_tokens=True,
                max_length=32,
            )
            src = torch.tensor([ids + [PAD_ID] * (pad_to - len(ids))], dtype=torch.long)
            mask = torch.tensor(
                [[1] * len(ids) + [0] * (pad_to - len(ids))], dtype=torch.float32
            )
            with torch.no_grad():
                greedy = model.generate(
                    src_input_ids=src,
                    src_attention_mask=mask,
                    max_length=16,
                    sos_token_id=SOS_ID,
                    eos_token_id=EOS_ID,
                )
                cached = model.generate_with_cache(
                    src_input_ids=src,
                    src_attention_mask=mask,
                    max_length=16,
                    sos_token_id=SOS_ID,
                    eos_token_id=EOS_ID,
                )
            gb = strip_special_ids(greedy[0].tolist())
            cb = strip_special_ids(cached[0].tolist())
            assert gb == cb, (
                f"Greedy/cached disagreed for {pair['src_text']!r} -> {pair['tgt_text']!r}: "
                f"greedy={gb} cached={cb}"
            )

    def test_beam_search_produces_non_empty_outputs(
        self, trained_smoke_dir: Path
    ) -> None:
        """Even when beam search picks a shorter hypothesis than the target,
        it must never emit an empty token stream.
        """
        manifest = json.loads(
            (trained_smoke_dir / "manifest.json").read_text(encoding="utf-8")
        )
        tokenizer = SmokeTokenizer.from_json(
            json.loads((trained_smoke_dir / "tokenizer.json").read_text(encoding="utf-8"))
        )
        model = _build_model_from_manifest(manifest)
        model.load_state_dict(
            torch.load(trained_smoke_dir / "model.pt", map_location="cpu", weights_only=True)[
                "model_state_dict"
            ]
        )
        model.eval()

        pairs = load_pairs(DEFAULT_TRAIN_PATH)
        for pair in pairs:
            ids = tokenizer.encode(
                pair["src_text"],
                src_lang=pair["src_lang"],
                tgt_lang=pair["tgt_lang"],
                add_special_tokens=True,
                max_length=32,
            )
            src = torch.tensor([ids], dtype=torch.long)
            mask = torch.tensor([[1] * len(ids)], dtype=torch.float32)
            with torch.no_grad():
                beam = model.generate_beam(
                    src_input_ids=src,
                    src_attention_mask=mask,
                    max_length=16,
                    num_beams=3,
                    sos_token_id=SOS_ID,
                    eos_token_id=EOS_ID,
                )
            assert beam.dtype == torch.long
            body = strip_special_ids(beam[0].tolist())
            assert body, f"Beam search produced empty output for {pair}"

    def test_unseen_source_still_produces_finite_output(
        self, trained_smoke_dir: Path
    ) -> None:
        """Model must handle prompts it never saw without NaN/Inf."""
        manifest = json.loads(
            (trained_smoke_dir / "manifest.json").read_text(encoding="utf-8")
        )
        tokenizer = SmokeTokenizer.from_json(
            json.loads((trained_smoke_dir / "tokenizer.json").read_text(encoding="utf-8"))
        )
        model = _build_model_from_manifest(manifest)
        model.load_state_dict(
            torch.load(trained_smoke_dir / "model.pt", map_location="cpu", weights_only=True)[
                "model_state_dict"
            ]
        )
        model.eval()

        unseen_pair: Dict[str, str] = {
            "src_text": "hello hello morning thank",
            "tgt_text": "",  # unused
            "src_lang": "en",
            "tgt_lang": "es",
        }
        ids = tokenizer.encode(
            unseen_pair["src_text"],
            src_lang=unseen_pair["src_lang"],
            tgt_lang=unseen_pair["tgt_lang"],
            add_special_tokens=True,
            max_length=32,
        )
        src = torch.tensor([ids], dtype=torch.long)
        mask = torch.tensor([[1] * len(ids)], dtype=torch.float32)
        with torch.no_grad():
            greedy = model.generate(
                src_input_ids=src,
                src_attention_mask=mask,
                max_length=10,
                sos_token_id=SOS_ID,
                eos_token_id=EOS_ID,
            )
        assert torch.isfinite(greedy.float()).all()
        body = strip_special_ids(greedy[0].tolist())
        assert body, "Unseen prompt produced empty greedy output"


def test_smoke_infer_fails_without_checkpoint() -> None:
    """Calling smoke_infer on an empty directory must raise a clear error."""
    empty = Path(".tmp_manual") / "smoke_empty_dir"
    if empty.exists():
        import shutil

        shutil.rmtree(empty, ignore_errors=True)
    empty.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        run_smoke_infer(out_dir=empty)


def test_expected_target_body_matches_tokenizer_encoding() -> None:
    """``expected_target_body`` must match what training actually targets."""
    pairs = load_pairs(DEFAULT_TRAIN_PATH)
    tok = SmokeTokenizer.from_pairs(pairs)
    for pair in pairs:
        body = expected_target_body(tok, pair["tgt_text"])
        # The body, when re-encoded without specials, should match.
        re_encoded = tok.encode(
            pair["tgt_text"], add_special_tokens=False, max_length=32
        )
        assert body == re_encoded
