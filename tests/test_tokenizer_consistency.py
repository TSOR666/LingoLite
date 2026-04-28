"""End-to-end tokenizer consistency tests with real SentencePiece artifacts.

The other test module (`test_translation_tokenizer.py`) covers unit-level
behaviour against a mocked SentencePiece processor. This file complements
that with integration tests that train a real SentencePiece model on the
tiny multilingual dataset and check the properties the training/eval
pipeline actually relies on:

* encode -> decode round-trips for each supported language pair
* language-pair symmetry (changing only ``tgt_lang`` should change only the
  trailing ``<lang>`` control token, never the body)
* save -> ``from_pretrained`` produces a tokenizer that encodes identically
* truncation preserves the ``</s> <tgt> <lang>`` suffix at the byte level
* special-token IDs are stable across save/load
* batch encoding shape consistency (used by the dataloader collate path)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest
import sentencepiece as spm  # type: ignore[import-untyped]
import torch

from lingolite.translation_tokenizer import TranslationTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SUPPORTED_LANGS: Tuple[str, ...] = ("en", "es", "fr", "de", "it", "da")
_TINY_DATASET = Path("examples/data/tiny_dataset.json")


def _build_corpus(pairs: List[Dict[str, str]], out_path: Path) -> None:
    """Write src+tgt sentences (one per line) for SentencePiece training."""
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(pair["src_text"].strip() + "\n")
            fh.write(pair["tgt_text"].strip() + "\n")


def _train_sp_model(corpus_path: Path, model_prefix: Path, vocab_size: int, languages: List[str]) -> None:
    """Train SentencePiece with absolute paths so the CWD stays clean."""
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,  # tiny corpus; cover everything
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


@pytest.fixture(scope="session")
def trained_tokenizer() -> Iterator[Tuple[TranslationTokenizer, Path]]:
    """Train a real SentencePiece tokenizer once for the whole test session.

    Returns the loaded tokenizer plus the directory holding the saved
    artifacts so individual tests can re-load it via ``from_pretrained``.

    Uses a project-local ``.tmp_pytest`` root rather than pytest's tmp factory
    because the latter scans ``%TEMP%/pytest-of-<user>`` which on Windows often
    contains stale, permission-denied directories from earlier test runs.
    """
    if not _TINY_DATASET.exists():
        pytest.skip(f"missing tiny dataset at {_TINY_DATASET}")

    pairs = json.loads(_TINY_DATASET.read_text(encoding="utf-8"))

    project_tmp_root = Path(".tmp_pytest")
    project_tmp_root.mkdir(exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="real_spm_tokenizer_", dir=str(project_tmp_root)))
    corpus_path = out_dir / "corpus.txt"
    _build_corpus(pairs, corpus_path)

    model_prefix = out_dir / "translation_tokenizer"
    languages = list(_SUPPORTED_LANGS)
    # Vocab needs room for: pad/sos/eos/unk (4) + src/tgt (2) + langs (6)
    # plus subword pieces from the corpus. SentencePiece refuses a vocab_size
    # larger than the unique-piece count it can produce, which on this 39-pair
    # tiny corpus tops out around ~109. 100 leaves a safety margin while still
    # exercising real subwording.
    sp_vocab_size = 100
    _train_sp_model(corpus_path, model_prefix, vocab_size=sp_vocab_size, languages=languages)

    tokenizer = TranslationTokenizer(
        languages=languages, vocab_size=sp_vocab_size, model_prefix="translation_tokenizer"
    )
    tokenizer.load(str(model_prefix.with_suffix(".model")))

    # Persist a tokenizer_config.json so ``from_pretrained`` can re-load it.
    save_dir = out_dir / "saved"
    save_dir.mkdir(exist_ok=True)
    (save_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "languages": languages,
                "vocab_size": sp_vocab_size,
                "special_tokens": tokenizer.special_tokens,
                "model_prefix": "translation_tokenizer",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(model_prefix.with_suffix(".model"), save_dir / "translation_tokenizer.model")
    shutil.copyfile(model_prefix.with_suffix(".vocab"), save_dir / "translation_tokenizer.vocab")

    try:
        yield tokenizer, save_dir
    finally:
        # Best-effort cleanup; ignore Windows permission quirks on stale handles.
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Special-token stability
# ---------------------------------------------------------------------------

class TestSpecialTokenStability:
    """Special-token IDs must be the values the rest of the pipeline assumes."""

    def test_reserved_special_token_ids(self, trained_tokenizer: Tuple[TranslationTokenizer, Path]) -> None:
        tokenizer, _ = trained_tokenizer
        assert tokenizer.token_to_id["<pad>"] == 0
        assert tokenizer.token_to_id["<s>"] == 1
        assert tokenizer.token_to_id["</s>"] == 2
        assert tokenizer.token_to_id["<unk>"] == 3

    def test_every_supported_language_has_a_token(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        for lang in _SUPPORTED_LANGS:
            assert f"<{lang}>" in tokenizer.token_to_id, f"missing language token for {lang!r}"

    def test_src_and_tgt_control_tokens_resolve(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        assert "<src>" in tokenizer.token_to_id
        assert "<tgt>" in tokenizer.token_to_id


# ---------------------------------------------------------------------------
# Encode / decode round-trips
# ---------------------------------------------------------------------------

class TestEncodeDecodeRoundTrip:
    """Real SentencePiece round-trips have to recover the original text body."""

    @pytest.mark.parametrize("text", ["Hello world", "Good morning", "Thank you"])
    def test_roundtrip_preserves_text_without_specials(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path], text: str
    ) -> None:
        tokenizer, _ = trained_tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        # SentencePiece may normalise whitespace; compare on lowercased non-space chars.
        assert "".join(decoded.lower().split()) == "".join(text.lower().split())

    def test_translation_format_round_trip(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        text = "Good morning"
        ids = tokenizer.encode(text, src_lang="en", tgt_lang="es", add_special_tokens=True)

        # Translation framing: <src> <en> ... </s> <tgt> <es>
        assert ids[0] == tokenizer.token_to_id["<src>"]
        assert ids[1] == tokenizer.token_to_id["<en>"]
        assert ids[-3] == tokenizer.token_to_id["</s>"]
        assert ids[-2] == tokenizer.token_to_id["<tgt>"]
        assert ids[-1] == tokenizer.token_to_id["<es>"]

        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert "".join(decoded.lower().split()) == "".join(text.lower().split())

    def test_decode_strips_padding_and_specials(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        body = tokenizer.encode("Hello", add_special_tokens=False)
        # Pretend this came back from generate(): SOS + body + EOS + PAD * k.
        padded = (
            [tokenizer.token_to_id["<s>"]]
            + body
            + [tokenizer.token_to_id["</s>"]]
            + [tokenizer.token_to_id["<pad>"]] * 4
        )
        decoded = tokenizer.decode(padded, skip_special_tokens=True)
        assert "<pad>" not in decoded
        assert "<s>" not in decoded
        assert "</s>" not in decoded
        assert "hello" in decoded.lower()


# ---------------------------------------------------------------------------
# Language-pair consistency
# ---------------------------------------------------------------------------

class TestLanguagePairConsistency:
    """The body tokens must be independent of the requested target language."""

    def test_changing_target_language_only_changes_trailing_token(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        text = "Thank you"
        en_es = tokenizer.encode(text, src_lang="en", tgt_lang="es", add_special_tokens=True)
        en_fr = tokenizer.encode(text, src_lang="en", tgt_lang="fr", add_special_tokens=True)

        # Same length and same prefix/body; only the very last token differs.
        assert len(en_es) == len(en_fr)
        assert en_es[:-1] == en_fr[:-1]
        assert en_es[-1] == tokenizer.token_to_id["<es>"]
        assert en_fr[-1] == tokenizer.token_to_id["<fr>"]

    def test_changing_source_language_only_changes_second_token(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        text = "Hello"
        en_de = tokenizer.encode(text, src_lang="en", tgt_lang="de", add_special_tokens=True)
        es_de = tokenizer.encode(text, src_lang="es", tgt_lang="de", add_special_tokens=True)

        assert len(en_de) == len(es_de)
        assert en_de[0] == es_de[0]  # both <src>
        assert en_de[1] == tokenizer.token_to_id["<en>"]
        assert es_de[1] == tokenizer.token_to_id["<es>"]
        assert en_de[2:] == es_de[2:]  # body + suffix unchanged

    @pytest.mark.parametrize(
        "src_lang,tgt_lang",
        [
            ("en", "es"),
            ("en", "fr"),
            ("en", "de"),
            ("en", "it"),
            ("en", "da"),
            ("es", "en"),
        ],
    )
    def test_every_dataset_pair_encodes(
        self,
        trained_tokenizer: Tuple[TranslationTokenizer, Path],
        src_lang: str,
        tgt_lang: str,
    ) -> None:
        tokenizer, _ = trained_tokenizer
        ids = tokenizer.encode("Hello", src_lang=src_lang, tgt_lang=tgt_lang, add_special_tokens=True)
        # Minimum framed length: <src> <lang> <body>=1+ </s> <tgt> <lang>
        assert len(ids) >= 6


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    """Real-model truncation must keep the trailing control tokens intact."""

    def test_translation_truncation_keeps_suffix(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        # Force a long body by repeating a known phrase.
        long_text = " ".join(["Hello world"] * 30)

        max_length = 12
        ids = tokenizer.encode(
            long_text,
            src_lang="en",
            tgt_lang="fr",
            add_special_tokens=True,
            max_length=max_length,
        )

        assert len(ids) == max_length
        # Prefix preserved
        assert ids[0] == tokenizer.token_to_id["<src>"]
        assert ids[1] == tokenizer.token_to_id["<en>"]
        # Suffix preserved (this is the property the pre-existing fix added)
        assert ids[-3] == tokenizer.token_to_id["</s>"]
        assert ids[-2] == tokenizer.token_to_id["<tgt>"]
        assert ids[-1] == tokenizer.token_to_id["<fr>"]

    def test_truncation_below_minimum_raises(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        with pytest.raises(ValueError, match="max_length must be at least"):
            tokenizer.encode(
                "anything",
                src_lang="en",
                tgt_lang="es",
                add_special_tokens=True,
                max_length=4,  # need at least 5 (prefix=2 + suffix=3)
            )


# ---------------------------------------------------------------------------
# Save / load consistency
# ---------------------------------------------------------------------------

class TestSaveLoadConsistency:
    """A reloaded tokenizer must be byte-identical for the same input."""

    def test_from_pretrained_matches_in_memory(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, save_dir = trained_tokenizer
        reloaded = TranslationTokenizer.from_pretrained(save_dir)

        for text in ["Hello", "Good morning", "Thank you"]:
            assert tokenizer.encode(text, add_special_tokens=False) == reloaded.encode(
                text, add_special_tokens=False
            ), f"mismatch on plain encode for {text!r}"
            assert tokenizer.encode(
                text, src_lang="en", tgt_lang="es", add_special_tokens=True
            ) == reloaded.encode(
                text, src_lang="en", tgt_lang="es", add_special_tokens=True
            ), f"mismatch on translation-format encode for {text!r}"

    def test_special_token_ids_survive_reload(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, save_dir = trained_tokenizer
        reloaded = TranslationTokenizer.from_pretrained(save_dir)

        for tok in ["<pad>", "<s>", "</s>", "<unk>", "<src>", "<tgt>"]:
            assert tokenizer.token_to_id[tok] == reloaded.token_to_id[tok]
        for lang in _SUPPORTED_LANGS:
            assert tokenizer.token_to_id[f"<{lang}>"] == reloaded.token_to_id[f"<{lang}>"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same input must produce the same output every call - no hidden state."""

    def test_repeated_encode_is_deterministic(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        first = tokenizer.encode("Hello world", src_lang="en", tgt_lang="es")
        for _ in range(3):
            assert tokenizer.encode("Hello world", src_lang="en", tgt_lang="es") == first

    def test_batch_encode_matches_single_encode(
        self, trained_tokenizer: Tuple[TranslationTokenizer, Path]
    ) -> None:
        tokenizer, _ = trained_tokenizer
        texts = ["Hello", "Good morning"]
        per_text = [
            tokenizer.encode(t, src_lang="en", tgt_lang="es", add_special_tokens=True)
            for t in texts
        ]

        batched = tokenizer.batch_encode(texts, src_lang="en", tgt_lang="es", padding=True)
        input_ids = batched["input_ids"]
        attention_mask = batched["attention_mask"]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)

        max_len = max(len(ids) for ids in per_text)
        assert input_ids.shape == (2, max_len)
        assert attention_mask.shape == (2, max_len)

        pad_id = tokenizer.token_to_id["<pad>"]
        for row, ids in zip(input_ids.tolist(), per_text):
            assert row[: len(ids)] == ids
            # Anything beyond ``len(ids)`` must be padding only.
            assert all(t == pad_id for t in row[len(ids):])
