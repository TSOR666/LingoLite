"""Regression coverage for bounded and multilingual training batches."""

import pytest
import torch
from torch.utils.data import DataLoader

from lingolite.mobile_translation_model import create_model
from lingolite.tokenizer_stub import StubTranslationTokenizer
from lingolite.training import (
    TranslationBatchSampler, TranslationDataset, TranslationTrainer,
    build_arg_parser, collate_fn,
)


def _dataset(lengths, pretokenize=False):
    tokenizer = StubTranslationTokenizer(languages=['en', 'es'])
    data = [
        {'src_text': 'word ' * length, 'tgt_text': 'palabra ' * (length // 2 + 1),
         'src_lang': 'en', 'tgt_lang': 'es'}
        for length in lengths
    ]
    return TranslationDataset(data, tokenizer, pretokenize=pretokenize)


def _flatten(sampler):
    return [index for batch in sampler for index in batch]


def test_token_budget_bounds_actual_padded_tensors_without_dropping_examples():
    dataset = _dataset([1, 20, 4, 40, 9, 3, 70, 8, 2], pretokenize=True)
    sampler = TranslationBatchSampler(dataset, batch_size=4, max_tokens=120)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    assert sorted(_flatten(sampler)) == list(range(len(dataset)))
    assert len(loader) == len(list(sampler))
    for batch in loader:
        assert batch['src_input_ids'].shape[0] <= 4
        assert batch['src_input_ids'].numel() + batch['tgt_input_ids'].numel() <= 120
        assert torch.equal(batch['src_attention_mask'].bool(), batch['src_input_ids'].ne(0))
        assert torch.equal(batch['tgt_attention_mask'].bool(), batch['tgt_input_ids'].ne(0))


def test_bucketing_reduces_padding_for_mixed_length_corpus():
    dataset = _dataset([1, 90] * 40)
    sampler = TranslationBatchSampler(dataset, batch_size=8, seed=42)
    lengths = dataset.sequence_lengths()

    def padded_cost(batches):
        return sum(len(batch) * (max(lengths[i][0] for i in batch) +
                                 max(lengths[i][1] for i in batch)) for batch in batches)

    random_order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
    naive_batches = [random_order[i:i + 8] for i in range(0, len(random_order), 8)]
    assert padded_cost(sampler) < padded_cost(naive_batches) * 0.7


def test_epoch_plans_reproduce_and_reshuffle_without_using_global_rng():
    dataset = _dataset([4] * 100)
    sampler = TranslationBatchSampler(dataset, batch_size=8, seed=10)
    initial = list(sampler)
    torch.manual_seed(1000)
    assert list(sampler) == initial
    sampler.set_epoch(1)
    assert list(sampler) != initial
    assert sorted(_flatten(sampler)) == list(range(100))
    sampler.set_epoch(0)
    assert list(sampler) == initial
    initial[0].clear()
    assert len(_flatten(sampler)) == 100


def test_temperature_sampling_increases_low_resource_direction_exposure():
    dataset = _dataset([3] * 1000)
    for item in dataset.data[900:]:
        item['src_lang'], item['tgt_lang'] = 'es', 'en'
    corpus_sampler = TranslationBatchSampler(dataset, language_temperature=1.0, seed=123)
    balanced_sampler = TranslationBatchSampler(dataset, language_temperature=2.0, seed=123)
    baseline = sum(index >= 900 for index in _flatten(corpus_sampler))
    balanced = sum(index >= 900 for index in _flatten(balanced_sampler))

    # Corpus probabilities are 90:10; T=2 changes them to 75:25.
    assert 60 < baseline < 140
    assert 200 < balanced < 300
    assert len(_flatten(balanced_sampler)) == len(dataset)
    assert list(balanced_sampler) == list(TranslationBatchSampler(
        dataset, language_temperature=2.0, seed=123))


@pytest.mark.parametrize('pretokenize', [False, True])
def test_length_planning_encodes_once_and_returns_independent_metadata(pretokenize, monkeypatch):
    dataset = _dataset([1, 2, 3], pretokenize=pretokenize)
    calls = []
    original = dataset.tokenizer.encode

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset.tokenizer, 'encode', tracked)
    first = dataset.sequence_lengths()
    second = dataset.sequence_lengths()
    assert first == second
    assert len(calls) == (0 if pretokenize else 2 * len(dataset))
    first.clear()
    assert len(dataset.sequence_lengths()) == len(dataset)


@pytest.mark.parametrize('kwargs', [
    {'batch_size': 0}, {'batch_size': True}, {'max_tokens': 0},
    {'max_tokens': 3.5}, {'bucket_size': 0},
    {'language_temperature': 0}, {'language_temperature': float('nan')},
    {'language_temperature': float('inf')},
])
def test_invalid_batch_controls_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        TranslationBatchSampler(_dataset([1]), **kwargs)


def test_oversized_example_is_rejected_instead_of_exceeding_budget():
    with pytest.raises(ValueError, match='Example 0.*max_tokens=5'):
        TranslationBatchSampler(_dataset([10]), max_tokens=5)


def test_empty_dataset_has_no_batches():
    sampler = TranslationBatchSampler(_dataset([]), max_tokens=64, language_temperature=2)
    assert len(sampler) == 0
    assert list(sampler) == []


def test_training_cli_keeps_new_batching_opt_in():
    parser = build_arg_parser()
    defaults = parser.parse_args(['--train-data', 'train.json', '--tokenizer-path', 'tokenizer'])
    assert defaults.max_tokens is None
    assert defaults.language_temperature is None
    options = parser.parse_args([
        '--train-data', 'train.json', '--tokenizer-path', 'tokenizer',
        '--max-tokens', '4096', '--language-temperature', '2',
    ])
    assert options.max_tokens == 4096
    assert options.language_temperature == 2


def test_trainer_runs_token_budget_batches_and_advances_epoch(tmp_path):
    dataset = _dataset([1, 5], pretokenize=True)
    sampler = TranslationBatchSampler(dataset, batch_size=2, max_tokens=16)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    model = create_model(
        vocab_size=dataset.tokenizer.get_vocab_size(), model_size='tiny',
        d_model=32, n_encoder_layers=1, n_decoder_layers=1,
        n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=32, dropout=0.0,
    )
    trainer = TranslationTrainer(
        model, loader, device='cpu', max_steps=4, save_dir=str(tmp_path),
    )
    trainer.train(num_epochs=2)
    assert trainer.global_step == 4
    assert sampler.epoch == 1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
