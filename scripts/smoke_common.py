"""Shared helpers for the CPU smoke train/infer pipeline.

The smoke scripts don't ship a trained SentencePiece model — instead they use
a deterministic ``SmokeTokenizer`` that encodes each whitespace token as a
fixed vocabulary slot. That is enough to exercise the full train -> save ->
load -> decode pipeline on CPU without pulling in real training corpora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Reserved IDs shared across the smoke tokenizer and dataset.
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SRC_ID = 4
TGT_ID = 5

SUPPORTED_LANGS: Tuple[str, ...] = ("en", "es", "fr", "de", "it", "da")
# Lang IDs occupy slots 6..(6 + len(SUPPORTED_LANGS) - 1) to stay aligned with
# the rest of the project's tokenizer layout.
LANG_IDS: Dict[str, int] = {lang: 6 + i for i, lang in enumerate(SUPPORTED_LANGS)}
# First free ID after reserved + language tokens.
_FIRST_FREE_ID = 6 + len(SUPPORTED_LANGS)

DEFAULT_TRAIN_PATH = Path("examples/data/tiny_smoke_dataset.json")
DEFAULT_FULL_TRAIN_PATH = Path("examples/data/tiny_dataset.json")
DEFAULT_VAL_PATH = Path("examples/data/tiny_dataset_val.json")


def load_pairs(path: Path = DEFAULT_TRAIN_PATH) -> List[Dict[str, str]]:
    """Load translation pairs from the tiny-dataset JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Smoke dataset not found at {path}. Run 'python scripts/make_tiny_dataset.py' first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list of translation pairs")
    return data


@dataclass
class SmokeTokenizer:
    """Deterministic whitespace-level tokenizer for the smoke pipeline.

    The tokenizer is built from a corpus of source/target texts: each unique
    whitespace-separated token is assigned a stable ID. Encoding follows the
    same wrap-with-special-tokens convention as ``TranslationTokenizer``:

        encode("hello world", "en", "es") =
            [<src>, <en>, "hello", "world", </s>, <tgt>, <es>]

    The tokenizer is intentionally minimal -- it doesn't learn BPE and does no
    normalization beyond case-folding -- so the smoke test can exactly check
    round-tripping of training examples after an overfit run.
    """

    word_to_id: Dict[str, int]
    id_to_word: Dict[int, str]
    languages: List[str]
    pad_token_id: int = PAD_ID
    sos_token_id: int = SOS_ID
    eos_token_id: int = EOS_ID
    unk_token_id: int = UNK_ID

    # Properties for parity with TranslationTokenizer's API where used.
    @property
    def token_to_id(self) -> Dict[str, int]:
        """Return a mapping compatible with ``TranslationTokenizer.token_to_id``.

        Includes the special / language tokens so dataset code that looks up
        ``tokenizer.token_to_id["<es>"]`` continues to work.
        """
        table = dict(self.word_to_id)
        table.update(
            {
                "<pad>": PAD_ID,
                "<s>": SOS_ID,
                "</s>": EOS_ID,
                "<unk>": UNK_ID,
                "<src>": SRC_ID,
                "<tgt>": TGT_ID,
            }
        )
        for lang in self.languages:
            table[f"<{lang}>"] = LANG_IDS[lang]
        return table

    def get_vocab_size(self) -> int:
        return _FIRST_FREE_ID + len(self.word_to_id)

    @classmethod
    def from_pairs(
        cls, pairs: List[Dict[str, str]], languages: Optional[List[str]] = None
    ) -> "SmokeTokenizer":
        langs = list(languages or SUPPORTED_LANGS)
        # Collect unique words in insertion order so the vocabulary is
        # deterministic across runs (Python 3.7+ dict preserves order).
        word_to_id: Dict[str, int] = {}
        next_id = _FIRST_FREE_ID
        for pair in pairs:
            for text in (pair["src_text"], pair["tgt_text"]):
                for tok in text.lower().split():
                    if tok not in word_to_id:
                        word_to_id[tok] = next_id
                        next_id += 1
        id_to_word = {i: w for w, i in word_to_id.items()}
        return cls(word_to_id=word_to_id, id_to_word=id_to_word, languages=langs)

    def _body_ids(self, text: str, max_length: int) -> List[int]:
        body: List[int] = []
        for tok in text.lower().split():
            body.append(self.word_to_id.get(tok, UNK_ID))
        # Leave room for up to 4 wrapper tokens (<src>/<lang>, </s>, <tgt>/<lang>).
        budget = max(1, max_length - 4)
        return body[:budget]

    def encode(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: int = 32,
    ) -> List[int]:
        body = self._body_ids(text, max_length)
        if add_special_tokens and src_lang and tgt_lang:
            if src_lang not in self.languages or tgt_lang not in self.languages:
                raise ValueError(
                    f"Unsupported language pair ({src_lang}->{tgt_lang}); supported: {self.languages}"
                )
            tokens = [
                SRC_ID,
                LANG_IDS[src_lang],
                *body,
                EOS_ID,
                TGT_ID,
                LANG_IDS[tgt_lang],
            ]
            return tokens[:max_length]
        if add_special_tokens:
            return [SOS_ID, *body, EOS_ID][:max_length]
        return body[:max_length]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        specials = {PAD_ID, SOS_ID, EOS_ID, UNK_ID, SRC_ID, TGT_ID, *LANG_IDS.values()}
        words = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special_tokens and tid in specials:
                continue
            words.append(self.id_to_word.get(tid, "<unk>"))
        return " ".join(words)

    def to_json(self) -> Dict[str, object]:
        """Serialize the tokenizer to a JSON-compatible dict for checkpointing."""
        return {
            "word_to_id": self.word_to_id,
            "languages": self.languages,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "SmokeTokenizer":
        word_to_id = {str(k): int(v) for k, v in data["word_to_id"].items()}  # type: ignore[index]
        id_to_word = {i: w for w, i in word_to_id.items()}
        languages = list(data.get("languages", list(SUPPORTED_LANGS)))  # type: ignore[arg-type]
        return cls(word_to_id=word_to_id, id_to_word=id_to_word, languages=languages)


def strip_special_ids(token_ids: List[int]) -> List[int]:
    """Remove control / language IDs from a decoded sequence."""
    specials = {PAD_ID, SOS_ID, EOS_ID, UNK_ID, SRC_ID, TGT_ID, *LANG_IDS.values()}
    return [int(t) for t in token_ids if int(t) not in specials]


def expected_target_body(tokenizer: SmokeTokenizer, tgt_text: str) -> List[int]:
    """Return the body (no specials) that training targets for ``tgt_text``."""
    return [tokenizer.word_to_id.get(tok, UNK_ID) for tok in tgt_text.lower().split()]
