"""
Example Usage of Mobile Translation Model
Demonstrates tokenizer training, model creation, and inference
"""

import torch
from pathlib import Path

from lingolite.translation_tokenizer import TranslationTokenizer
from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationDataset, TranslationTrainer, collate_fn
from torch.utils.data import DataLoader


def example_tokenizer_training():
    """Example: Train a tokenizer on multilingual corpus."""
    print("=" * 80)
    print("EXAMPLE 1: TOKENIZER TRAINING")
    print("=" * 80)
    
    # Create tokenizer
    tokenizer = TranslationTokenizer(
        languages=['en', 'es', 'fr', 'de', 'it', 'da'],
        vocab_size=24000,
        model_prefix='translation_tokenizer'
    )
    
    # In practice, you would train on real corpus files
    # tokenizer.train([
    #     'data/corpus_en.txt',
    #     'data/corpus_es.txt',
    #     'data/corpus_fr.txt',
    #     'data/corpus_de.txt',
    #     'data/corpus_it.txt',
    # ])
    
    print("âœ“ Tokenizer created")
    print(f"  Languages: {tokenizer.languages}")
    print(f"  Target vocab size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {len(tokenizer.special_tokens)}")
    
    # Example encoding/decoding (with trained model)
    # text = "Hello, how are you?"
    # tokens = tokenizer.encode(text, src_lang='en', tgt_lang='es')
    # decoded = tokenizer.decode(tokens)
    
    print("\nNote: To use tokenizer, first train it on your corpus:")
    print("  tokenizer.train(['corpus1.txt', 'corpus2.txt', ...])")


def example_model_creation():
    """Example: Create models of different sizes."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: MODEL CREATION")
    print("=" * 80)
    
    vocab_size = 24000
    
    print("\nAvailable model sizes:")
    print("-" * 80)
    
    for size in ['tiny', 'small', 'medium']:
        model = create_model(vocab_size=vocab_size, model_size=size)
        params = model.count_parameters()
        
        # Calculate sizes
        fp32_mb = params['total'] * 4 / (1024**2)
        int8_mb = params['total'] * 1 / (1024**2)
        
        print(f"\n{size.upper()}:")
        print(f"  Parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
        print(f"  FP32 size: {fp32_mb:.1f} MB")
        print(f"  INT8 size: {int8_mb:.1f} MB")
        print(f"  Encoder layers: {model.encoder.layers.__len__()}")
        print(f"  Decoder layers: {model.decoder.layers.__len__()}")


def example_inference():
    """Example: Use model for translation inference."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: TRANSLATION INFERENCE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    vocab_size = 24000
    model = create_model(vocab_size=vocab_size, model_size='small')
    model = model.to(device)
    model.eval()
    
    print("âœ“ Model created and loaded")
    
    # In practice, you would:
    # 1. Load trained tokenizer
    # tokenizer = TranslationTokenizer.from_pretrained('./tokenizer')
    
    # 2. Tokenize input
    # text = "Hello, how are you?"
    # encoded = tokenizer.batch_encode([text], src_lang='en', tgt_lang='es')
    # src_ids = encoded['input_ids'].to(device)
    # src_mask = encoded['attention_mask'].to(device)
    
    # 3. Generate translation
    # generated = model.generate(
    #     src_input_ids=src_ids,
    #     src_attention_mask=src_mask,
    #     max_length=128,
    #     sos_token_id=tokenizer.sos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    # )
    
    # 4. Decode output
    # translation = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    # print(f"Translation: {translation}")
    
    # Simulated inference
    batch = 2
    src_len = 32
    src_ids = torch.randint(0, vocab_size, (batch, src_len)).to(device)
    src_mask = torch.ones(batch, src_len).to(device)
    
    with torch.no_grad():
        generated = model.generate(
            src_input_ids=src_ids,
            src_attention_mask=src_mask,
            max_length=50,
            sos_token_id=1,
            eos_token_id=2,
        )
    
    print(f"âœ“ Generation successful")
    print(f"  Input shape: {src_ids.shape}")
    print(f"  Generated shape: {generated.shape}")


def example_training_setup():
    """Example: Set up training pipeline."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: TRAINING SETUP")
    print("=" * 80)
    
    # This is a conceptual example showing the training pipeline
    
    print("\nTraining Pipeline:")
    print("-" * 80)
    
    print("\n1. Prepare Data:")
    print("   # Load parallel corpus")
    print("   data = [")
    print("       {'src_text': 'Hello', 'tgt_text': 'Hola', 'src_lang': 'en', 'tgt_lang': 'es'},")
    print("       # ... more examples")
    print("   ]")
    
    print("\n2. Create Dataset:")
    print("   tokenizer = TranslationTokenizer.from_pretrained('./tokenizer')")
    print("   dataset = TranslationDataset(data, tokenizer, max_length=128)")
    print("   train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)")
    
    print("\n3. Create Model:")
    print("   model = create_model(vocab_size=24000, model_size='small')")
    
    print("\n4. Create Trainer:")
    print("   trainer = TranslationTrainer(")
    print("       model=model,")
    print("       train_loader=train_loader,")
    print("       val_loader=val_loader,")
    print("       learning_rate=3e-4,")
    print("       device='cuda'")
    print("   )")
    
    print("\n5. Train:")
    print("   trainer.train(num_epochs=10)")
    
    print("\n6. Save Model:")
    print("   torch.save(model.state_dict(), 'translation_model.pt')")


def example_quantization():
    """Example: Quantize model for mobile deployment."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: MODEL QUANTIZATION")
    print("=" * 80)
    
    device = torch.device('cpu')  # Quantization typically done on CPU
    
    # Create model
    vocab_size = 24000
    model = create_model(vocab_size=vocab_size, model_size='small')
    model = model.to(device)
    
    print("âœ“ Original model created")
    params = model.count_parameters()
    fp32_size = params['total'] * 4 / (1024**2)
    print(f"  FP32 size: {fp32_size:.1f} MB")
    
    # Quantization (conceptual - requires additional setup)
    print("\nQuantization Steps:")
    print("-" * 80)
    
    print("\n1. Dynamic Quantization (simplest):")
    print("   quantized_model = torch.quantization.quantize_dynamic(")
    print("       model,")
    print("       {torch.nn.Linear},")
    print("       dtype=torch.qint8")
    print("   )")
    print("   # ~4x size reduction, ~2-3x speedup")
    
    print("\n2. Quantization-Aware Training (best quality):")
    print("   model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')")
    print("   model_prepared = torch.quantization.prepare_qat(model)")
    print("   # Train for a few epochs")
    print("   model_quantized = torch.quantization.convert(model_prepared)")
    
    print("\n3. Export for Mobile:")
    print("   # TorchScript")
    print("   traced = torch.jit.trace(model_quantized, example_inputs)")
    print("   traced.save('model_mobile.pt')")
    print("")
    print("   # ONNX")
    print("   torch.onnx.export(model, example_inputs, 'model.onnx')")
    print("")
    print("   # TensorFlow Lite")
    print("   # Convert ONNX -> TF -> TFLite")
    
    int8_size = params['total'] * 1 / (1024**2)
    print(f"\nâœ“ Expected INT8 size: {int8_size:.1f} MB (~4x reduction)")


def example_complete_workflow():
    """Example: Complete workflow from data to deployment."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: COMPLETE WORKFLOW")
    print("=" * 80)
    
    print("\nðŸ“‹ STEP-BY-STEP GUIDE")
    print("=" * 80)
    
    print("\n1ï¸âƒ£  DATA PREPARATION")
    print("-" * 80)
    print("   â€¢ Collect parallel corpus (e.g., from OPUS, CCMatrix)")
    print("   â€¢ Format: List of {'src_text', 'tgt_text', 'src_lang', 'tgt_lang'}")
    print("   â€¢ Recommended: 1M+ sentence pairs per language pair")
    
    print("\n2ï¸âƒ£  TOKENIZER TRAINING")
    print("-" * 80)
    print("   tokenizer = TranslationTokenizer(languages=['en', 'es', 'fr'])")
    print("   tokenizer.train(['corpus_en.txt', 'corpus_es.txt', 'corpus_fr.txt'])")
    print("   tokenizer.save('./tokenizer')")
    
    print("\n3ï¸âƒ£  MODEL TRAINING")
    print("-" * 80)
    print("   model = create_model(vocab_size=24000, model_size='small')")
    print("   trainer = TranslationTrainer(model, train_loader, val_loader)")
    print("   trainer.train(num_epochs=10)")
    print("   # Takes: 1-2 weeks on single GPU for 'small' model")
    
    print("\n4ï¸âƒ£  EVALUATION")
    print("-" * 80)
    print("   # Compute BLEU score on test set")
    print("   from sacrebleu import corpus_bleu")
    print("   translations = [model.translate(src) for src in test_sources]")
    print("   bleu = corpus_bleu(translations, [test_references])")
    
    print("\n5ï¸âƒ£  QUANTIZATION")
    print("-" * 80)
    print("   quantized_model = quantize_model(model)")
    print("   # Size: ~60MB INT8 vs ~240MB FP32")
    
    print("\n6ï¸âƒ£  EXPORT FOR MOBILE")
    print("-" * 80)
    print("   # iOS/Android")
    print("   traced = torch.jit.trace(quantized_model, example_input)")
    print("   traced.save('translation_model_mobile.pt')")
    
    print("\n7ï¸âƒ£  MOBILE APP INTEGRATION")
    print("-" * 80)
    print("   // iOS (Swift)")
    print("   let model = try! TorchModule(fileAtPath: modelPath)")
    print("   let output = model.predict(input)")
    print("")
    print("   // Android (Kotlin)")
    print("   val module = LiteModuleLoader.load(modelPath)")
    print("   val output = module.forward(IValue.from(input))")
    
    print("\nâœ“ End-to-End Pipeline Complete!")


if __name__ == "__main__":
    # Run all examples
    example_tokenizer_training()
    example_model_creation()
    example_inference()
    example_training_setup()
    example_quantization()
    example_complete_workflow()
    
    print("\n" + "=" * 80)
    print("ðŸ“š For more details, see README.md")
    print("=" * 80)
