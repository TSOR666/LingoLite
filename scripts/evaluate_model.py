"""
BLEU Model Evaluation Script
Automatic translation quality measurement using sacrebleu
Evaluates trained LingoLite models on test datasets
"""

import torch
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import json
from tqdm import tqdm

try:
    import sacrebleu
except ImportError:
    print("ERROR: sacrebleu not installed. Install with: pip install sacrebleu")
    exit(1)

from lingolite.mobile_translation_model import MobileTranslationModel
from lingolite.translation_tokenizer import TranslationTokenizer
from lingolite.utils import logger


def load_parallel_data(
    source_file: Path,
    target_file: Path,
    max_samples: Optional[int] = None
) -> tuple[List[str], List[str]]:
    """
    Load parallel source and target sentences.

    Args:
        source_file: Path to source language file (one sentence per line)
        target_file: Path to target language file (one sentence per line)
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        source_sentences: List of source sentences
        target_sentences: List of target (reference) sentences
    """
    with open(source_file, 'r', encoding='utf-8') as f:
        source_sentences = [line.strip() for line in f]

    with open(target_file, 'r', encoding='utf-8') as f:
        target_sentences = [line.strip() for line in f]

    if len(source_sentences) != len(target_sentences):
        raise ValueError(
            f"Source and target files have different lengths: "
            f"{len(source_sentences)} vs {len(target_sentences)}"
        )

    if max_samples is not None:
        source_sentences = source_sentences[:max_samples]
        target_sentences = target_sentences[:max_samples]

    logger.info(f"Loaded {len(source_sentences)} parallel sentences")
    return source_sentences, target_sentences


def translate_batch(
    model: MobileTranslationModel,
    tokenizer: TranslationTokenizer,
    source_sentences: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
    use_cache: bool = True,
) -> List[str]:
    """
    Translate a batch of sentences.

    Args:
        model: Translation model
        tokenizer: Tokenizer
        source_sentences: List of source sentences
        device: Device to run on
        batch_size: Batch size for translation
        max_length: Maximum generation length
        use_cache: Whether to use KV cache for faster generation

    Returns:
        translations: List of translated sentences
    """
    model.eval()
    translations = []

    with torch.no_grad():
        for i in tqdm(range(0, len(source_sentences), batch_size), desc="Translating"):
            batch = source_sentences[i:i + batch_size]

            # Tokenize
            src_ids, src_mask = tokenizer.encode_batch(
                batch,
                max_length=max_length,
                return_tensors=True
            )
            src_ids = src_ids.to(device)
            src_mask = src_mask.to(device)

            # Generate translations
            if use_cache:
                generated = model.generate_with_cache(
                    src_input_ids=src_ids,
                    src_attention_mask=src_mask,
                    max_length=max_length,
                    sos_token_id=tokenizer.sos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            else:
                generated = model.generate(
                    src_input_ids=src_ids,
                    src_attention_mask=src_mask,
                    max_length=max_length,
                    sos_token_id=tokenizer.sos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode
            batch_translations = tokenizer.decode_batch(generated)
            translations.extend(batch_translations)

    return translations


def compute_bleu_metrics(
    hypotheses: List[str],
    references: List[List[str]],
    lowercase: bool = False,
    tokenize: str = '13a',
) -> Dict[str, float]:
    """
    Compute BLEU score using sacrebleu.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations (can be multiple refs per hyp)
        lowercase: Whether to lowercase before scoring
        tokenize: Tokenization method ('13a', 'intl', 'zh', 'ja-mecab', etc.)

    Returns:
        Dictionary with BLEU scores and metrics
    """
    # sacrebleu expects references as list of lists (one list per reference set)
    # If we have single references per hypothesis, wrap each in a list
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    # Transpose references if needed by checking orientation
    # Per-sentence format: len(references) == len(hypotheses), each inner list has multiple refs
    # Per-reference-set format: len(references) == num_refs, each inner list has all sentences
    if len(references[0]) == 1:
        # Single reference: [[ref1], [ref2], ...] -> [[ref1, ref2, ...]]
        references = [[ref[0] for ref in references]]
    elif len(references) == len(hypotheses):
        # Multiple references in per-sentence format: transpose to per-reference-set
        # [[ref1_s1, ref2_s1], [ref1_s2, ref2_s2], ...] -> [[ref1_s1, ref1_s2, ...], [ref2_s1, ref2_s2, ...]]
        num_refs = len(references[0])
        references = [[sent[i] for sent in references] for i in range(num_refs)]
    # else: already in per-reference-set format, no transpose needed

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        references,
        lowercase=lowercase,
        tokenize=tokenize,
    )

    return {
        'bleu': bleu.score,
        'bleu_1': bleu.precisions[0],
        'bleu_2': bleu.precisions[1],
        'bleu_3': bleu.precisions[2],
        'bleu_4': bleu.precisions[3],
        'bp': bleu.bp,  # Brevity penalty
        'sys_len': bleu.sys_len,
        'ref_len': bleu.ref_len,
    }


def compute_chrf(
    hypotheses: List[str],
    references: List[List[str]],
) -> float:
    """
    Compute chrF score (character n-gram F-score).
    More robust than BLEU for morphologically rich languages.

    Args:
        hypotheses: List of hypothesis translations
        references: List of reference translations

    Returns:
        chrF score
    """
    if isinstance(references[0], str):
        references = [[ref] for ref in references]

    # Transpose references if needed by checking orientation
    # Per-sentence format: len(references) == len(hypotheses), each inner list has multiple refs
    # Per-reference-set format: len(references) == num_refs, each inner list has all sentences
    if len(references[0]) == 1:
        # Single reference: [[ref1], [ref2], ...] -> [[ref1, ref2, ...]]
        references = [[ref[0] for ref in references]]
    elif len(references) == len(hypotheses):
        # Multiple references in per-sentence format: transpose to per-reference-set
        # [[ref1_s1, ref2_s1], [ref1_s2, ref2_s2], ...] -> [[ref1_s1, ref1_s2, ...], [ref2_s1, ref2_s2, ...]]
        num_refs = len(references[0])
        references = [[sent[i] for sent in references] for i in range(num_refs)]
    # else: already in per-reference-set format, no transpose needed

    chrf = sacrebleu.corpus_chrf(hypotheses, references)
    return chrf.score


def evaluate_model(
    model_path: Path,
    tokenizer_path: Path,
    source_file: Path,
    target_file: Path,
    output_file: Optional[Path] = None,
    batch_size: int = 32,
    max_length: int = 128,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    device: Optional[str] = None,
    save_translations: bool = False,
) -> Dict[str, float]:
    """
    Evaluate translation model with BLEU score.

    Args:
        model_path: Path to saved model checkpoint
        tokenizer_path: Path to tokenizer directory
        source_file: Path to source sentences
        target_file: Path to reference translations
        output_file: Optional path to save evaluation results
        batch_size: Batch size for translation
        max_length: Maximum generation length
        max_samples: Maximum number of samples to evaluate (None = all)
        use_cache: Whether to use KV cache
        device: Device to run on (cuda/cpu)
        save_translations: Whether to save translations to file

    Returns:
        Dictionary with evaluation metrics
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = TranslationTokenizer.load(tokenizer_path)

    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = MobileTranslationModel(**config)
    else:
        # Assume default config
        model = MobileTranslationModel(
            vocab_size=tokenizer.vocab_size,
            d_model=512,
            n_encoder_layers=6,
            n_decoder_layers=6,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model.count_parameters()['total']:,} parameters")

    # Load parallel data
    source_sentences, reference_sentences = load_parallel_data(
        source_file, target_file, max_samples
    )

    # Translate
    logger.info("Starting translation...")
    translations = translate_batch(
        model=model,
        tokenizer=tokenizer,
        source_sentences=source_sentences,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_cache=use_cache,
    )

    # Compute BLEU
    logger.info("Computing BLEU score...")
    bleu_metrics = compute_bleu_metrics(translations, reference_sentences)

    # Compute chrF
    logger.info("Computing chrF score...")
    chrf_score = compute_chrf(translations, reference_sentences)

    # Combine metrics
    metrics = {
        **bleu_metrics,
        'chrf': chrf_score,
        'num_samples': len(translations),
    }

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"BLEU Score:      {metrics['bleu']:.2f}")
    print(f"chrF Score:      {metrics['chrf']:.2f}")
    print(f"BLEU-1:          {metrics['bleu_1']:.2f}")
    print(f"BLEU-2:          {metrics['bleu_2']:.2f}")
    print(f"BLEU-3:          {metrics['bleu_3']:.2f}")
    print(f"BLEU-4:          {metrics['bleu_4']:.2f}")
    print(f"Brevity Penalty: {metrics['bp']:.4f}")
    print(f"Samples:         {metrics['num_samples']}")
    print("=" * 80 + "\n")

    # Save results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    # Save translations if requested
    if save_translations:
        trans_file = output_file.parent / f"{output_file.stem}_translations.txt"
        with open(trans_file, 'w', encoding='utf-8') as f:
            for trans in translations:
                f.write(trans + '\n')
        logger.info(f"Translations saved to {trans_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation model with BLEU")
    parser.add_argument('--model', type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument('--tokenizer', type=Path, required=True, help="Path to tokenizer directory")
    parser.add_argument('--source', type=Path, required=True, help="Path to source sentences file")
    parser.add_argument('--target', type=Path, required=True, help="Path to reference translations file")
    parser.add_argument('--output', type=Path, help="Path to save evaluation results (JSON)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for translation")
    parser.add_argument('--max-length', type=int, default=128, help="Maximum generation length")
    parser.add_argument('--max-samples', type=int, help="Maximum number of samples to evaluate")
    parser.add_argument('--no-cache', action='store_true', help="Disable KV cache")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help="Device to use")
    parser.add_argument('--save-translations', action='store_true', help="Save translations to file")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        source_file=args.source,
        target_file=args.target,
        output_file=args.output,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
        use_cache=not args.no_cache,
        device=args.device,
        save_translations=args.save_translations,
    )


if __name__ == '__main__':
    main()
