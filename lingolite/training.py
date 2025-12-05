"""
Training Script for Mobile Translation Model
Complete pipeline from data to trained model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import json
import argparse
import sys
import logging
from tqdm import tqdm
import random

from .mobile_translation_model import MobileTranslationModel, create_model
from .translation_tokenizer import TranslationTokenizer
from .utils import setup_logger

# Set up logger
logger = setup_logger(name="lingolite_training", level=logging.INFO)


class TranslationDataset(Dataset):
    """
    Dataset for translation pairs.
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: TranslationTokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            data: List of dicts with 'src_text', 'tgt_text', 'src_lang', 'tgt_lang'
            tokenizer: TranslationTokenizer instance
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # Validate required fields
        required_fields = ['src_text', 'tgt_text', 'src_lang', 'tgt_lang']
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            raise ValueError(
                f"Dataset item at index {idx} is missing required fields: {missing_fields}. "
                f"Expected format: {{'src_text': str, 'tgt_text': str, 'src_lang': str, 'tgt_lang': str}}"
            )

        # Tokenize source
        src_ids = self.tokenizer.encode(
            text=item['src_text'],
            src_lang=item['src_lang'],
            tgt_lang=item['tgt_lang'],
            add_special_tokens=True,
            max_length=self.max_length,
        )

        # Tokenize target (format: <tgt_lang> text </s>)
        tgt_text = item['tgt_text']
        tgt_ids = self.tokenizer.encode(
            text=tgt_text,
            add_special_tokens=False,
            max_length=self.max_length,
        )
        
        # Add language token and EOS
        if item['tgt_lang'] not in self.tokenizer.languages:
            raise ValueError(f"Target language '{item['tgt_lang']}' not supported. Supported: {self.tokenizer.languages}")
        tgt_lang_id = self.tokenizer.token_to_id[f"<{item['tgt_lang']}>"]
        tgt_ids = [tgt_lang_id] + tgt_ids + [self.tokenizer.eos_token_id]
        
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate batch with padding.
    
    Args:
        batch: List of samples from dataset
        pad_token_id: ID for padding token
    
    Returns:
        Batched tensors with padding
    """
    # Handle empty batch
    if len(batch) == 0:
        return {
            'src_input_ids': torch.empty(0, 0, dtype=torch.long),
            'tgt_input_ids': torch.empty(0, 0, dtype=torch.long),
            'src_attention_mask': torch.empty(0, 0, dtype=torch.float),
            'tgt_attention_mask': torch.empty(0, 0, dtype=torch.float),
        }
    
    # Find max lengths
    max_src_len = max(len(item['src_ids']) for item in batch)
    max_tgt_len = max(len(item['tgt_ids']) for item in batch)
    
    # Pad sequences
    src_ids = []
    tgt_ids = []
    src_mask = []
    tgt_mask = []
    
    for item in batch:
        # Pad source
        src = item['src_ids']
        src_padding = [pad_token_id] * (max_src_len - len(src))
        src_ids.append(src + src_padding)
        src_mask.append([1] * len(src) + [0] * len(src_padding))
        
        # Pad target
        tgt = item['tgt_ids']
        tgt_padding = [pad_token_id] * (max_tgt_len - len(tgt))
        tgt_ids.append(tgt + tgt_padding)
        tgt_mask.append([1] * len(tgt) + [0] * len(tgt_padding))
    
    return {
        'src_input_ids': torch.tensor(src_ids, dtype=torch.long),
        'tgt_input_ids': torch.tensor(tgt_ids, dtype=torch.long),
        'src_attention_mask': torch.tensor(src_mask, dtype=torch.float),
        'tgt_attention_mask': torch.tensor(tgt_mask, dtype=torch.float),
    }


class TranslationTrainer:
    """
    Trainer for translation model.
    """
    
    def __init__(
        self,
        model: MobileTranslationModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.1,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
    ):
        """
        Args:
            model: Translation model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            gradient_clip: Gradient clipping value
            label_smoothing: Label smoothing factor
            device: Device to train on
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Validate max_steps
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        self.max_steps = max_steps

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler (FIXED)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=max_steps,
            pct_start=warmup_steps / max_steps,
            anneal_strategy='cos',
        )

        self.gradient_clip = gradient_clip
        self.label_smoothing = label_smoothing

        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
        
        Returns:
            loss: Training loss
            metrics: Dictionary of metrics
        """
        # Handle empty batch
        if batch['src_input_ids'].numel() == 0:
            return 0.0, {'loss': 0.0, 'grad_norm': 0.0, 'lr': 0.0}
        
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        loss = self.model.compute_loss(
            src_input_ids=batch['src_input_ids'],
            tgt_input_ids=batch['tgt_input_ids'],
            src_attention_mask=batch['src_attention_mask'],
            tgt_attention_mask=batch['tgt_attention_mask'],
            label_smoothing=self.label_smoothing,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.gradient_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        metrics = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': self.scheduler.get_last_lr()[0],
        }
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Skip empty batches
            if batch['src_input_ids'].numel() == 0:
                continue
                
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Compute loss
            loss = self.model.compute_loss(
                src_input_ids=batch['src_input_ids'],
                tgt_input_ids=batch['tgt_input_ids'],
                src_attention_mask=batch['src_attention_mask'],
                tgt_attention_mask=batch['tgt_attention_mask'],
                label_smoothing=0.0,  # No smoothing for validation
            )
            
            # Count non-padding tokens
            num_tokens = batch['tgt_attention_mask'][:, 1:].sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / max(1e-8, total_tokens)
        perplexity = float(np.exp(avg_loss))
        
        return {
            'val_loss': avg_loss,
            'perplexity': perplexity,
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Checkpoint loaded: {load_path}")
    
    def train(
        self,
        num_epochs: int,
        eval_steps: int = 5000,
        save_steps: int = 10000,
        log_steps: int = 100,
    ):
        """
        Main training loop.

        Args:
            num_epochs: Number of training epochs
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            log_steps: Log metrics every N steps
        """
        print("=" * 80)
        print("TRAINING MOBILE TRANSLATION MODEL")
        print("=" * 80)
        print(f"Max steps: {self.max_steps}")

        running_loss = 0
        training_complete = False

        for epoch in range(num_epochs):
            if training_complete:
                break

            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                # Check if we've reached max_steps BEFORE taking a step
                if self.global_step >= self.max_steps:
                    print(f"\n✓ Reached max_steps ({self.max_steps})")
                    training_complete = True
                    break

                loss, metrics = self.train_step(batch)
                running_loss += loss

                # Update progress bar
                if self.global_step % log_steps == 0:
                    avg_loss = running_loss / log_steps
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{metrics['lr']:.2e}",
                        'step': self.global_step,
                    })
                    running_loss = 0

                # Evaluation
                if self.global_step % eval_steps == 0 and self.val_loader is not None:
                    val_metrics = self.evaluate()
                    print(f"\nStep {self.global_step} - Validation:")
                    print(f"  Loss: {val_metrics['val_loss']:.4f}")
                    print(f"  Perplexity: {val_metrics['perplexity']:.2f}")

                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best_model.pt')

                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        print(f"\n{'='*80}")
        print("✓ TRAINING COMPLETE")
        print(f"  Final step: {self.global_step}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}")


# Command-line training entry point
def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the training entry point."""

    parser = argparse.ArgumentParser(description='Train LingoLite Translation Model')

    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data JSON file (list of dicts with src_text, tgt_text, src_lang, tgt_lang)')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation data JSON file')
    parser.add_argument('--tokenizer-path', type=str, required=True,
                        help='Path to trained tokenizer directory')

    # Model arguments
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size preset')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='Vocabulary size (auto-detected from tokenizer if not specified)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=2000,
                        help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')

    # Logging arguments
    parser.add_argument('--log-steps', type=int, default=100,
                        help='Log metrics every N steps')
    parser.add_argument('--eval-steps', type=int, default=5000,
                        help='Evaluate every N steps')
    parser.add_argument('--save-steps', type=int, default=10000,
                        help='Save checkpoint every N steps')

    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line interface for model training."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Validate required files exist
    train_data_path = Path(args.train_data)
    if not train_data_path.exists():
        logger.error(f"Training data not found at {args.train_data}")
        sys.exit(1)

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at {args.tokenizer_path}")
        print("\nTo train a tokenizer, use:")
        print("  from lingolite.translation_tokenizer import TranslationTokenizer")
        print("  tokenizer = TranslationTokenizer(languages=['en', 'es', 'fr'])")
        print("  tokenizer.train(['corpus1.txt', 'corpus2.txt'])")
        print("  tokenizer.save('./tokenizer')")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("LINGOLITE TRAINING")
    logger.info("=" * 80)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = TranslationTokenizer.from_pretrained(str(tokenizer_path))
    vocab_size = args.vocab_size or tokenizer.get_vocab_size()

    # Validate vocab_size
    if vocab_size <= 0:
        logger.error(f"Invalid vocabulary size: {vocab_size}. Tokenizer may not be properly initialized.")
        sys.exit(1)

    logger.info(f"Tokenizer loaded: vocab_size={vocab_size}")

    # Load training data
    logger.info(f"Loading training data from {args.train_data}...")
    with open(args.train_data, 'r') as f:
        train_data = json.load(f)
    logger.info(f"Loaded {len(train_data)} training examples")

    # Load validation data
    val_loader = None
    if args.val_data:
        print(f"\nLoading validation data from {args.val_data}...")
        with open(args.val_data, 'r') as f:
            val_data = json.load(f)
        print(f"✓ Loaded {len(val_data)} validation examples")

        val_dataset = TranslationDataset(val_data, tokenizer, max_length=args.max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
        )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TranslationDataset(train_data, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )
    print(f"✓ Train loader: {len(train_loader)} batches")
    if val_loader:
        print(f"✓ Val loader: {len(val_loader)} batches")

    # Select device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nDevice: {device}")

    # Create model
    print(f"\nCreating model (size={args.model_size})...")
    model = create_model(vocab_size=vocab_size, model_size=args.model_size)
    params = model.count_parameters()
    print(f"✓ Model created: {params['total']/1e6:.1f}M parameters")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = TranslationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_clip=args.gradient_clip,
        label_smoothing=args.label_smoothing,
        device=device,
        save_dir=args.save_dir,
    )
    print("✓ Trainer ready")

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    try:
        trainer.train(
            num_epochs=args.num_epochs,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            log_steps=args.log_steps,
        )
        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("⚠ TRAINING INTERRUPTED")
        print("=" * 80)
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pt')
        print("✓ Checkpoint saved")
        sys.exit(1)
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("✗ TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
