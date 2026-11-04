"""Transport-independent, checkpoint-compatible batched translation."""

from dataclasses import dataclass
import math
import threading
from typing import Any, List, Sequence, Union

import torch


@dataclass(frozen=True)
class GenerationOptions:
    max_length: int = 128
    method: str = "greedy"
    num_beams: int = 4
    temperature: float = 1.0
    length_penalty: float = 1.0
    min_new_tokens: int = 0
    no_repeat_ngram_size: int = 0

    def __post_init__(self) -> None:
        if self.method not in {"greedy", "beam", "fast"}:
            raise ValueError("Unknown generation method")
        for name, minimum in (("max_length", 2), ("num_beams", 1),
                              ("min_new_tokens", 0), ("no_repeat_ngram_size", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.min_new_tokens >= self.max_length:
            raise ValueError("min_new_tokens must be smaller than max_length")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.length_penalty) or self.length_penalty < 0:
            raise ValueError("length_penalty must be finite and non-negative")


@dataclass(frozen=True)
class TranslationInput:
    text: str
    src_lang: str = "en"
    tgt_lang: str = "es"


@dataclass(frozen=True)
class TranslationResult:
    translation: str
    input_length: int
    output_length: int


class TranslationEngine:
    """Own one model replica; serialize its mutable decode state across callers.

    Tokenization, padding, inference and decoding all run in the calling thread.
    Async transports should invoke this through a bounded worker, as in serving.py.
    """

    def __init__(self, model: Any, tokenizer: Any, device: torch.device,
                 max_source_length: int = 512) -> None:
        if max_source_length < 1:
            raise ValueError("max_source_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        config = model.get_config()
        self.max_sequence_length = int(config.get("max_seq_len", 512))
        self.max_source_length = min(max_source_length, self.max_sequence_length)
        if getattr(model, "vocab_size", tokenizer.get_vocab_size()) != tokenizer.get_vocab_size():
            raise ValueError("Model and tokenizer vocabulary sizes differ")
        if getattr(model, "pad_token_id", tokenizer.pad_token_id) != tokenizer.pad_token_id:
            raise ValueError("Model and tokenizer padding token IDs differ")
        special_ids = [tokenizer.pad_token_id, tokenizer.sos_token_id, tokenizer.eos_token_id]
        if len(set(special_ids)) != 3 or any(not 0 <= token < tokenizer.get_vocab_size() for token in special_ids):
            raise ValueError("Tokenizer special token IDs must be distinct and within the vocabulary")
        self._lock = threading.Lock()

    def token_cost(self, options: GenerationOptions) -> int:
        """Conservative per-item padded source + beam-expanded decode budget."""
        beams = options.num_beams if options.method == "beam" else 1
        return self.max_source_length + options.max_length * beams

    def translate_batch(
        self, inputs: Sequence[TranslationInput], options: GenerationOptions,
    ) -> List[Union[TranslationResult, ValueError]]:
        """Translate together, isolating invalid inputs from neighboring requests.

        A generation failure raises for the entire batch. Input errors occupy their
        original result position, so a bad request never fails unrelated clients.
        """
        with self._lock, torch.inference_mode():
            return self._translate_batch(inputs, options)

    def _translate_batch(
        self, inputs: Sequence[TranslationInput], options: GenerationOptions,
    ) -> List[Union[TranslationResult, ValueError]]:
        if options.max_length > self.max_sequence_length:
            raise ValueError(f"max_length exceeds model limit ({self.max_sequence_length})")
        results: List[Union[TranslationResult, ValueError]] = []
        encoded = []
        positions = []
        for item in inputs:
            try:
                if not item.text.strip():
                    raise ValueError("Text must contain non-whitespace characters")
                if item.src_lang not in self.tokenizer.languages or item.tgt_lang not in self.tokenizer.languages:
                    raise ValueError("Unsupported language code")
                # Do not truncate: silently discarding a clause changes its meaning.
                ids = self.tokenizer.encode(
                    item.text, src_lang=item.src_lang, tgt_lang=item.tgt_lang,
                    add_special_tokens=True, max_length=None,
                )
                if not ids or len(ids) > self.max_source_length:
                    raise ValueError(
                        f"Input has {len(ids)} tokens; supported limit is "
                        f"{self.max_source_length}. Split the text into shorter passages."
                    )
                positions.append(len(results))
                encoded.append(ids)
                results.append(ValueError("No translation generated"))
            except ValueError as exc:
                results.append(exc)
        if not encoded:
            return results

        width = max(map(len, encoded))
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.tensor(
            [ids + [pad_id] * (width - len(ids)) for ids in encoded],
            dtype=torch.long, device=self.device,
        )
        kwargs = dict(
            src_input_ids=input_ids, src_attention_mask=input_ids.ne(pad_id),
            max_length=options.max_length,
            sos_token_id=self.tokenizer.sos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            min_new_tokens=options.min_new_tokens,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
        )
        if options.method == "beam":
            output = self.model.generate_beam(
                **kwargs, num_beams=options.num_beams,
                length_penalty=options.length_penalty,
            )
        else:
            generate = self.model.generate_fast if options.method == "fast" else self.model.generate
            output = generate(**kwargs, temperature=options.temperature)
        rows = output.tolist()
        if len(rows) != len(encoded):
            raise RuntimeError("Model returned an unexpected batch size")
        for position, ids, row in zip(positions, encoded, rows):
            if self.tokenizer.eos_token_id in row:
                row = row[:row.index(self.tokenizer.eos_token_id) + 1]
            else:
                while row and row[-1] == pad_id:
                    row.pop()
            results[position] = TranslationResult(
                translation=self.tokenizer.decode(row, skip_special_tokens=True),
                input_length=len(ids), output_length=len(row),
            )
        return results
