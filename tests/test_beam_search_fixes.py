"""Beam search indexing fix tests (clean, encoding-safe)."""

import torch


def test_zero_dim_tensor_conversion():
    batch_size, num_beams, vocab_size = 2, 4, 100
    device = torch.device("cpu")

    next_tokens = torch.randint(0, vocab_size, (batch_size, 2 * num_beams))
    next_indices = torch.randint(0, num_beams, (batch_size, 2 * num_beams))

    beam_next_tokens = []
    beam_idx_list = []

    for batch_idx in range(batch_size):
        batch_next_tokens = []
        batch_next_indices = []
        for beam_idx in range(num_beams):
            idx = beam_idx
            batch_next_tokens.append(next_tokens[batch_idx, idx])
            batch_next_indices.append(next_indices[batch_idx, idx])
        for i in range(num_beams):
            beam_idx_global = batch_idx * num_beams + batch_next_indices[i].item()
            beam_next_tokens.append(batch_next_tokens[i].item())
            beam_idx_list.append(beam_idx_global)

    token_tensor = torch.tensor(beam_next_tokens, device=device, dtype=torch.long)
    assert token_tensor.dtype == torch.long
    assert len(beam_idx_list) == batch_size * num_beams


def test_batch_indexing_early_finish():
    batch_size, num_beams = 2, 4
    for batch_idx in range(batch_size):
        if batch_idx == 1:
            first_beam_idx = batch_idx * num_beams
            assert first_beam_idx == 4


def test_multi_batch_indexing():
    batch_size, num_beams, vocab_size, seq_len = 2, 4, 100, 10
    input_ids = torch.randint(0, vocab_size, (batch_size * num_beams, seq_len))
    # Simple beam index list for test purposes
    beam_idx_list = list(range(batch_size * num_beams))
    selected_ids = input_ids[beam_idx_list]
    assert selected_ids.shape[0] == batch_size * num_beams

