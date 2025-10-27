"""A lightweight tokenizer stub for development and demos.

This avoids requiring a trained SentencePiece model. It supports:
- Basic special tokens and language tokens
- Simple whitespace tokenization
- Stable ID assignment via hashing into a fixed vocab size

Intended for API/dev use only. Not suitable for training quality models.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional


class StubTranslationTokenizer:
    def __init__(self, languages: Optional[List[str]] = None, vocab_size: int = 2048):
        self.languages = languages or ["en", "es"]
        self.vocab_size = int(vocab_size)

        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.src_token = "<src>"
        self.tgt_token = "<tgt>"
        self.lang_tokens = [f"<{lang}>" for lang in self.languages]

        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
            self.src_token,
            self.tgt_token,
        ] + self.lang_tokens

        # Reserve low IDs for special tokens
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self._hash_offset = len(self.token_to_id)

    # Properties for API parity
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def sos_token_id(self) -> int:
        return self.token_to_id[self.sos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def _id_for_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        # Stable hash into [offset, vocab_size)
        m = hashlib.md5(tok.encode("utf-8")).hexdigest()
        hid = int(m, 16)
        idx = self._hash_offset + (hid % (self.vocab_size - self._hash_offset))
        # Cache reverse mapping for decode of seen tokens
        if idx not in self.id_to_token:
            self.id_to_token[idx] = tok
        return idx

    def encode(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        if add_special_tokens:
            toks: List[int] = []
            if src_lang and tgt_lang:
                if src_lang not in self.languages or tgt_lang not in self.languages:
                    raise ValueError("Unsupported language code for stub tokenizer")
                toks.extend([
                    self._id_for_token(self.src_token),
                    self._id_for_token(f"<{src_lang}>")
                ])
                for w in text.strip().split():
                    toks.append(self._id_for_token(w))
                toks.extend([
                    self._id_for_token(self.eos_token),
                    self._id_for_token(self.tgt_token),
                    self._id_for_token(f"<{tgt_lang}>")
                ])
            else:
                toks = [self.sos_token_id] + [self._id_for_token(w) for w in text.strip().split()] + [self.eos_token_id]
        else:
            toks = [self._id_for_token(w) for w in text.strip().split()]

        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        return toks

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in self.id_to_token and self.id_to_token[tid] in self.special_tokens:
                continue
            tok = self.id_to_token.get(tid, self.unk_token)
            # Do not emit special tokens in any case
            if tok in self.special_tokens:
                continue
            # Basic heuristic for lang tokens
            if tok.startswith("<") and tok.endswith(">"):
                continue
            words.append(tok)
        return " ".join(words)

