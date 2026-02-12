"""Synthetic smoke inference run for LingoLite."""

from __future__ import annotations

import torch

from lingolite.mobile_translation_model import create_model


def _assert_valid_tokens(name: str, tokens: torch.Tensor, max_length: int) -> None:
    if tokens.ndim != 2:
        raise RuntimeError(f"{name}: expected 2D output, got shape={tuple(tokens.shape)}")
    if tokens.shape[1] > max_length:
        raise RuntimeError(f"{name}: output length {tokens.shape[1]} exceeds max_length {max_length}")
    if not torch.isfinite(tokens.float()).all():
        raise RuntimeError(f"{name}: output contains NaN/Inf")


def main() -> None:
    torch.manual_seed(0)

    model = create_model(vocab_size=256, model_size="tiny", pad_token_id=0)
    model.eval()

    src_input_ids = torch.randint(0, 256, (2, 20))
    src_attention_mask = torch.ones(2, 20)
    max_length = 16

    with torch.no_grad():
        greedy = model.generate(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=1,
            eos_token_id=2,
        )
        _assert_valid_tokens("greedy", greedy, max_length)

        sampled = model.generate(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=1,
            eos_token_id=2,
            do_sample=True,
            top_k=16,
            top_p=0.9,
        )
        _assert_valid_tokens("sampled", sampled, max_length)

        cached = model.generate_fast(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=1,
            eos_token_id=2,
        )
        _assert_valid_tokens("cached", cached, max_length)

        beam = model.generate(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=1,
            eos_token_id=2,
            num_beams=2,
        )
        _assert_valid_tokens("beam", beam, max_length)

    print(
        "SMOKE INFER PASSED: "
        f"greedy={tuple(greedy.shape)} sampled={tuple(sampled.shape)} "
        f"cached={tuple(cached.shape)} beam={tuple(beam.shape)}"
    )


if __name__ == "__main__":
    main()
