"""
Quick verification test for the RoPE fix
"""
import sys
sys.path.insert(0, r'c:\Users\tsor\OneDrive - Danmarks Tekniske Universitet\Dokumenter\GitHub\LingoLite')

try:
    print("Testing imports...")
    from lingolite.model_components import RotaryPositionEmbedding
    print("✓ RotaryPositionEmbedding imported successfully")
    
    import torch
    print("\nTesting RoPE instantiation...")
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=512)
    print(f"✓ RoPE instance created: dim={rope.dim}, max_seq_len={rope.max_seq_len}")
    
    print("\nTesting RoPE forward pass...")
    q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq_len, dim)
    k = torch.randn(2, 8, 10, 64)
    q_rot, k_rot = rope(q, k)
    print(f"✓ Forward pass successful: q_rot.shape={q_rot.shape}, k_rot.shape={k_rot.shape}")
    
    print("\nTesting model creation...")
    from lingolite import create_model
    model = create_model(vocab_size=1000, model_size='tiny')
    print(f"✓ Model created successfully")
    
    params = model.count_parameters()
    print(f"✓ Model parameters: {params['total']:,}")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
