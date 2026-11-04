"""Opt-in translation decoding controls share one policy across all methods."""

import pytest
import torch
from torch import nn

from lingolite.generation_utils import GenerationConstraints
from lingolite.mobile_translation_model import create_model


class ConstantDecoder(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))

    def forward(self, input_ids, **kwargs):
        logits = self.scores.expand(input_ids.shape[0], 1, -1)
        return logits, None


def model_with_scores(scores):
    model = create_model(
        vocab_size=len(scores), model_size="tiny", d_model=16,
        n_encoder_layers=1, n_decoder_layers=1, n_heads=2, n_kv_heads=1,
        d_ff=32, max_seq_len=32, dropout=0.0,
    )
    model.decoder = ConstantDecoder(scores)
    return model


@pytest.mark.parametrize("method", ["generate", "generate_with_cache", "generate_fast", "generate_beam"])
def test_minimum_length_prevents_premature_eos_for_every_entry_point(method):
    model = model_with_scores([-20, -20, 10, 9, 0, 0, 0, 0])
    kwargs = {"num_beams": 2, "length_penalty": 0} if method == "generate_beam" else {}
    output = getattr(model, method)(
        torch.tensor([[1, 5, 2], [1, 6, 2]]), max_length=6,
        min_new_tokens=2, **kwargs,
    )
    assert output[:, :4].tolist() == [[1, 3, 3, 2], [1, 3, 3, 2]]
    assert model.training  # Generation restores the caller's mode.


@pytest.mark.parametrize("mode", ["greedy", "sampling", "beam"])
def test_repetition_and_special_token_controls_apply_to_all_decoding_modes(mode):
    model = model_with_scores([20, 19, 1, 10, 9, 8, 7, 6])
    kwargs = {"do_sample": True, "top_k": 1} if mode == "sampling" else {}
    if mode == "beam":
        kwargs = {"num_beams": 3}
    output = model.generate(
        torch.tensor([[1, 5, 2]]), max_length=10, min_new_tokens=4,
        no_repeat_ngram_size=2, suppress_tokens=[0, 1], **kwargs,
    )[0].tolist()[1:]
    if 2 in output:
        output = output[:output.index(2)]
    assert len(output) >= 4
    assert not {0, 1}.intersection(output)
    bigrams = list(zip(output, output[1:]))
    assert len(bigrams) == len(set(bigrams))


def test_ngram_mask_is_per_hypothesis_and_keeps_duplicate_matches():
    policy = GenerationConstraints(no_repeat_ngram_size=2)
    # Token 4 follows both matching and nonmatching prefixes in the first row.
    history = torch.tensor([[1, 3, 4, 5, 4, 3], [1, 3, 5, 4, 5, 3]])
    original = torch.zeros(2, 8)
    masked = policy.apply(original, history, eos_token_id=2)
    assert torch.isneginf(masked[0, 4])
    assert torch.isfinite(masked[0, 5])
    assert torch.isneginf(masked[1, 5])
    assert torch.isfinite(masked[1, 4])
    assert torch.isfinite(original).all()


def test_unigram_history_excludes_sos():
    masked = GenerationConstraints(no_repeat_ngram_size=1).apply(
        torch.zeros(1, 8), torch.tensor([[1, 3, 4]]), eos_token_id=2,
    )
    assert torch.isfinite(masked[0, 1])
    assert torch.isneginf(masked[0, 3:5]).all()


@pytest.mark.parametrize("kwargs", [
    {"min_new_tokens": -1}, {"min_new_tokens": 5},
    {"no_repeat_ngram_size": -1}, {"no_repeat_ngram_size": 1.5},
    {"suppress_tokens": [8]}, {"suppress_tokens": [-1]},
    {"suppress_tokens": [True]}, {"suppress_tokens": list(range(8))},
])
@pytest.mark.parametrize("method", ["generate", "generate_beam"])
def test_invalid_constraints_are_rejected_without_changing_training_mode(kwargs, method):
    model = model_with_scores([0] * 8)
    with pytest.raises(ValueError):
        getattr(model, method)(torch.tensor([[1, 3, 2]]), max_length=5, **kwargs)
    assert model.training


@pytest.mark.parametrize("method", ["generate", "generate_beam"])
def test_impossible_constraints_fail_instead_of_emitting_forbidden_tokens(method):
    model = model_with_scores([0] * 8)
    with pytest.raises(ValueError, match="no valid continuation"):
        getattr(model, method)(
            torch.tensor([[1, 3, 2]]), max_length=5, min_new_tokens=1,
            suppress_tokens=[0, 1, 3, 4, 5, 6, 7],
        )
    assert model.training


def test_disabled_constraints_preserve_default_generation():
    model = model_with_scores([-20, -20, 10, 9, 0, 0, 0, 0])
    source = torch.tensor([[1, 5, 2]])
    assert torch.equal(
        model.generate(source, max_length=6),
        model.generate(source, max_length=6, min_new_tokens=0,
                       no_repeat_ngram_size=0, suppress_tokens=[]),
    )
    assert model.generate(source, max_length=6).tolist() == [[1, 2]]


def test_beam_search_keeps_finished_translation_when_constraints_exhaust_live_beams():
    model = model_with_scores([0] * 8)
    output = model.generate_beam(
        torch.tensor([[1, 5, 2]]), max_length=6, num_beams=4,
        min_new_tokens=1, no_repeat_ngram_size=1,
        suppress_tokens=[0, 1, 4, 5, 6, 7],
    )
    assert output[0, :3].tolist() == [1, 3, 2]


@pytest.mark.parametrize("dtype,logit", [(torch.float32, -1e35), (torch.float16, -10000)])
def test_sampling_preserves_suppression_with_extreme_or_half_precision_logits(dtype, logit):
    model = model_with_scores([logit] * 8)
    model.decoder.to(dtype=dtype)
    output = model.generate(
        torch.tensor([[1, 5, 2]]), max_length=4, do_sample=True,
        suppress_tokens=[0, 1, 2, 4, 5, 6, 7],
    )
    assert output.tolist() == [[1, 3, 3, 3]]
