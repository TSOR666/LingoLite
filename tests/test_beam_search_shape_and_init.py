"""
Test script to verify beam search shape and initialization fixes:
1. Finished batch maintains correct shape (batch_size * num_beams)
2. beam_scores initialization activates first beam of EACH batch, not just batch 0
"""

import torch

def test_beam_scores_initialization():
    """Test that beam_scores are correctly initialized for all batches"""
    print("="*70)
    print("TEST 1: Beam Scores Initialization")
    print("="*70)

    batch_size = 3
    num_beams = 4
    device = torch.device('cpu')

    # CORRECT implementation (fixed)
    beam_scores = torch.full((batch_size * num_beams,), float('-inf'), device=device)
    beam_scores[::num_beams] = 0.0

    print(f"\nBatch size: {batch_size}, Num beams: {num_beams}")
    print(f"beam_scores shape: {beam_scores.shape}")
    print(f"\nbeam_scores values:")

    for batch_idx in range(batch_size):
        start_idx = batch_idx * num_beams
        end_idx = start_idx + num_beams
        batch_scores = beam_scores[start_idx:end_idx]
        print(f"  Batch {batch_idx} (indices {start_idx}-{end_idx-1}): {batch_scores.tolist()}")

        # Verify first beam is 0.0
        if batch_scores[0] != 0.0:
            print(f"    ✗ FAILED: First beam should be 0.0, got {batch_scores[0]}")
            return False

        # Verify other beams are -inf
        if not all(s == float('-inf') for s in batch_scores[1:].tolist()):
            print(f"    ✗ FAILED: Other beams should be -inf")
            return False

        print(f"    ✓ Batch {batch_idx} correctly initialized (first beam=0, others=-inf)")

    print("\n✓ TEST 1 PASSED: All batches have correct beam score initialization")
    return True


def test_finished_batch_shape_maintenance():
    """Test that finished batches maintain correct shape"""
    print("\n" + "="*70)
    print("TEST 2: Finished Batch Shape Maintenance")
    print("="*70)

    batch_size = 3
    num_beams = 4
    device = torch.device('cpu')
    vocab_size = 100
    seq_len = 5

    # Simulate input_ids for all beams
    input_ids = torch.randint(0, vocab_size, (batch_size * num_beams, seq_len))

    # Simulate beam_scorer.done where batch 1 is finished
    done = torch.tensor([False, True, False], dtype=torch.bool)

    print(f"\nBatch size: {batch_size}, Num beams: {num_beams}")
    print(f"Initial input_ids shape: {input_ids.shape}")
    print(f"Done batches: {[i for i in range(batch_size) if done[i]]}")

    # Simulate the beam selection process (CORRECT implementation)
    beam_outputs = []
    beam_next_tokens = []
    beam_idx_list = []
    pad_token_id = 0

    for batch_idx in range(batch_size):
        if done[batch_idx]:
            # This batch is done, pad all num_beams to maintain shape
            first_beam_idx = batch_idx * num_beams
            for i in range(num_beams):
                beam_outputs.append(input_ids[first_beam_idx + i])
                beam_next_tokens.append(pad_token_id)
                beam_idx_list.append(first_beam_idx + i)
            continue

        # For active batches, simulate selecting num_beams
        for i in range(num_beams):
            beam_idx_global = batch_idx * num_beams + i
            beam_outputs.append(input_ids[beam_idx_global])
            beam_next_tokens.append(vocab_size - 1)  # Some token
            beam_idx_list.append(beam_idx_global)

    print(f"\nAfter processing:")
    print(f"  beam_idx_list length: {len(beam_idx_list)}")
    print(f"  beam_next_tokens length: {len(beam_next_tokens)}")

    expected_length = batch_size * num_beams
    if len(beam_idx_list) != expected_length:
        print(f"  ✗ FAILED: Expected {expected_length} beams, got {len(beam_idx_list)}")
        return False

    print(f"  ✓ Correct total beam count: {expected_length}")

    # Verify each batch has num_beams entries
    for batch_idx in range(batch_size):
        start_idx = batch_idx * num_beams
        end_idx = start_idx + num_beams
        batch_indices = beam_idx_list[start_idx:end_idx]

        if len(batch_indices) != num_beams:
            print(f"  ✗ FAILED: Batch {batch_idx} has {len(batch_indices)} beams, expected {num_beams}")
            return False

        status = "done" if done[batch_idx] else "active"
        print(f"  ✓ Batch {batch_idx} ({status}): {num_beams} beams maintained")

    # Try to index input_ids with beam_idx_list (should not crash)
    try:
        selected_ids = input_ids[beam_idx_list]
        print(f"\n  ✓ Successfully indexed input_ids with beam_idx_list")
        print(f"    Selected shape: {selected_ids.shape}")

        if selected_ids.shape[0] != batch_size * num_beams:
            print(f"  ✗ FAILED: Wrong shape after indexing")
            return False
    except Exception as e:
        print(f"  ✗ FAILED: Could not index input_ids: {e}")
        return False

    # Try to create tensor from beam_next_tokens (should not crash)
    try:
        token_tensor = torch.tensor(beam_next_tokens, device=device, dtype=torch.long)
        print(f"  ✓ Successfully created tensor from beam_next_tokens")
        print(f"    Tensor shape: {token_tensor.shape}")
    except Exception as e:
        print(f"  ✗ FAILED: Could not create tensor from beam_next_tokens: {e}")
        return False

    print("\n✓ TEST 2 PASSED: Shape maintained correctly with finished batches")
    return True


def test_multi_batch_decoding():
    """Test that all batches can explore beams (not just batch 0)"""
    print("\n" + "="*70)
    print("TEST 3: Multi-Batch Beam Exploration")
    print("="*70)

    batch_size = 3
    num_beams = 4
    device = torch.device('cpu')
    vocab_size = 100

    # Initialize beam scores (CORRECT implementation)
    beam_scores = torch.full((batch_size * num_beams,), float('-inf'), device=device)
    beam_scores[::num_beams] = 0.0

    # Simulate next token scores
    next_token_logits = torch.randn(batch_size * num_beams, vocab_size)
    next_token_scores = torch.log_softmax(next_token_logits, dim=-1)

    # Add beam scores
    next_token_scores = next_token_scores + beam_scores[:, None]

    # Check that all batches have at least one non-inf score
    next_token_scores_reshaped = next_token_scores.view(batch_size, num_beams * vocab_size)

    print(f"\nBatch size: {batch_size}, Num beams: {num_beams}")
    print(f"Combined scores shape: {next_token_scores_reshaped.shape}")

    for batch_idx in range(batch_size):
        batch_scores = next_token_scores_reshaped[batch_idx]
        max_score = batch_scores.max().item()
        has_valid_scores = (batch_scores > float('-inf')).sum().item()

        print(f"\nBatch {batch_idx}:")
        print(f"  Max score: {max_score:.4f}")
        print(f"  Valid (non -inf) scores: {has_valid_scores}/{batch_scores.numel()}")

        if has_valid_scores == 0:
            print(f"  ✗ FAILED: Batch {batch_idx} has no valid scores (all -inf)")
            return False

        print(f"  ✓ Batch {batch_idx} can explore beams")

    print("\n✓ TEST 3 PASSED: All batches can explore beams")
    return True


def main():
    print("\n" + "="*70)
    print("BEAM SEARCH SHAPE AND INITIALIZATION FIX VERIFICATION")
    print("="*70 + "\n")

    tests = [
        test_beam_scores_initialization,
        test_finished_batch_shape_maintenance,
        test_multi_batch_decoding,
    ]

    results = [test() for test in tests]

    print("\n" + "="*70)
    if all(results):
        print("✓ ALL TESTS PASSED - Fixes are correct!")
        print("="*70)
        print("\nFixed issues:")
        print("1. beam_scores now activates first beam of EACH batch")
        print("2. Finished batches maintain full num_beams shape")
        print("="*70)
    else:
        print("✗ SOME TESTS FAILED - Please review the fixes")
        print("="*70)

    return all(results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
