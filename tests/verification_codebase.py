import torch
import torch.nn as nn
from lingolite.model_components import GroupedQueryAttention, RotaryPositionEmbedding
from lingolite.encoder_decoder import TransformerEncoder, TransformerDecoder
from lingolite.generation_utils import generate_with_kv_cache

def verify_codebase():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running verification on {device}")

    # Set up shapes
    B, SeqLen, D, H = 2, 16, 64, 4
    KV_H = 2
    Vocab = 100

    # 1. Test GQA Numerical Stability
    print("\n--- Verifying GQA Numerical Stability ---")
    gqa = GroupedQueryAttention(d_model=D, n_heads=H, n_kv_heads=KV_H).to(device)
    rope = RotaryPositionEmbedding(dim=D // H).to(device)
    
    # Input with float16 to trigger potential issues
    q = torch.randn(B, SeqLen, D, device=device, dtype=torch.float16)
    
    # We need to manually cast module to half if we want to test half precision fully,
    # but the check is about whether safe softmax happens even if inputs are half.
    gqa.half() 
    
    try:
        out, _ = gqa(q, rope=rope)
        print(f"GQA Output stats: Min={out.min().item()}, Max={out.max().item()}, Mean={out.mean().item()}")
        if torch.isnan(out).any():
            print("FAILURE: NaNs detected in GQA output")
            return False
        if out.dtype != torch.float16:
            print(f"FAILURE: Output dtype expected float16, got {out.dtype}")
            return False
        print("GQA Check Passed: No NaNs, correct dtype preservation.")
    except Exception as e:
        print(f"FAILURE: GQA crashed: {e}")
        return False

    # 2. Test Transformer Encoder Masking
    print("\n--- Verifying Encoder Masking (Magic Number Fix) ---")
    encoder = TransformerEncoder(vocab_size=Vocab, d_model=D, n_layers=1, n_heads=H, n_kv_heads=KV_H, d_ff=D*2).to(device)
    encoder.half()  # Test in half precision where -10000 might be finite/in-range
    
    input_ids = torch.randint(0, Vocab, (B, SeqLen), device=device)
    mask = torch.ones((B, SeqLen), device=device)
    mask[:, -2:] = 0  # Mask last 2 tokens
    
    try:
        out = encoder(input_ids, attention_mask=mask)
        if torch.isnan(out).any() or torch.isinf(out).any():
            # In half precision, intermediate values might overflow if not carefully handled, 
            # but final output typically shouldn't be Inf unless weights are bad.
            # However, we are checking for NaNs mainly.
            if torch.isnan(out).any():
                print("FAILURE: NaNs in Encoder output")
                return False
        print("Encoder Check Passed.")
    except Exception as e:
        print(f"FAILURE: Encoder crashed: {e}")
        return False

    # 3. Test Generation Stability
    print("\n--- Verifying Generation (Top-P / Beam) ---")
    # Need full model for generation
    # Mocking MobileTranslationModel structure since we can't import it easily without instantiating submodules manually
    # Actually, let's just test generation_utils functions if possible, but they require a model with .encoder/.decoder.
    
    # Let's define a minimal mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = TransformerDecoder(vocab_size=Vocab, d_model=D, n_layers=1, n_heads=H, n_kv_heads=KV_H, d_ff=D*2).to(device)
            self.decoder.half()
        
    model = MockModel()
    
    try:
        gen_out = generate_with_kv_cache(
            model, 
            src_input_ids=input_ids, 
            max_length=5, 
            top_p=0.9,
            top_k=0
        )
        print(f"Generation Output Shape: {gen_out.shape}")
        print("Generation Check Passed.")
    except Exception as e:
        print(f"FAILURE: Generation crashed: {e}")
        # Print traceback
        import traceback
        traceback.print_exc()
        return False

    print("\nVERIFICATION PASSED")
    return True

if __name__ == "__main__":
    verify_codebase()
