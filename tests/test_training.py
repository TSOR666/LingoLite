"""
Tests for training.py

Covers:
- TranslationDataset: data loading, indexing
- collate_fn: batching with padding
- TranslationTrainer: single step, gradients, checkpoints
- Loss computation: cross-entropy with label smoothing
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch
from typing import Dict, List
import tempfile
from pathlib import Path

from lingolite.mobile_translation_model import MobileTranslationModel, create_model
from lingolite.training import (
    TranslationDataset,
    TranslationTrainer,
    collate_fn,
)


# ============================================================================
# Mock Tokenizer for Testing
# ============================================================================

class MockTokenizer:
    """Mock tokenizer for testing without real SentencePiece model."""
    
    def __init__(self):
        self.languages = ['en', 'es']
        self.token_to_id = {
            '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3,
            '<src>': 4, '<tgt>': 5, '<en>': 6, '<es>': 7
        }
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
    
    def encode(self, text: str, src_lang: str = None, tgt_lang: str = None,
               add_special_tokens: bool = True, max_length: int = 128) -> List[int]:
        # Simple mock encoding: just return token IDs based on text length
        content = list(range(10, 10 + min(len(text.split()), max_length - 6)))
        if add_special_tokens:
            if src_lang and tgt_lang:
                # Translation format
                return [4, 6] + content + [2, 5, 7]
            else:
                return [1] + content + [2]
        return content
    
    def get_vocab_size(self) -> int:
        return 100


# ============================================================================
# TranslationDataset Tests
# ============================================================================

class TestTranslationDataset:
    """Tests for TranslationDataset."""

    @pytest.fixture
    def sample_data(self) -> List[Dict[str, str]]:
        """Create sample translation data."""
        return [
            {
                'src_text': 'Hello world',
                'tgt_text': 'Hola mundo',
                'src_lang': 'en',
                'tgt_lang': 'es'
            },
            {
                'src_text': 'Good morning',
                'tgt_text': 'Buenos días',
                'src_lang': 'en',
                'tgt_lang': 'es'
            },
            {
                'src_text': 'Thank you',
                'tgt_text': 'Gracias',
                'src_lang': 'en',
                'tgt_lang': 'es'
            },
        ]

    @pytest.fixture
    def mock_tokenizer(self) -> MockTokenizer:
        """Create mock tokenizer."""
        return MockTokenizer()

    def test_dataset_length(
        self, sample_data: List[Dict[str, str]], mock_tokenizer: MockTokenizer
    ) -> None:
        """Dataset should have correct length."""
        dataset = TranslationDataset(sample_data, mock_tokenizer)
        assert len(dataset) == 3

    def test_dataset_getitem(
        self, sample_data: List[Dict[str, str]], mock_tokenizer: MockTokenizer
    ) -> None:
        """Dataset should return correctly structured items."""
        dataset = TranslationDataset(sample_data, mock_tokenizer)
        item = dataset[0]
        
        assert 'src_input_ids' in item
        assert 'tgt_input_ids' in item
        assert 'src_attention_mask' in item
        assert 'tgt_attention_mask' in item

    def test_dataset_indexing(
        self, sample_data: List[Dict[str, str]], mock_tokenizer: MockTokenizer
    ) -> None:
        """Dataset should return different items for different indices."""
        dataset = TranslationDataset(sample_data, mock_tokenizer)
        
        item0 = dataset[0]
        item1 = dataset[1]
        
        # Items should be different (different source texts)
        assert item0['src_input_ids'] != item1['src_input_ids'] or \
               len(item0['src_input_ids']) != len(item1['src_input_ids'])

    def test_empty_dataset(self, mock_tokenizer: MockTokenizer) -> None:
        """Empty dataset should have length 0."""
        dataset = TranslationDataset([], mock_tokenizer)
        assert len(dataset) == 0


# ============================================================================
# collate_fn Tests
# ============================================================================

class TestCollateFn:
    """Tests for batch collation."""

    def test_pads_to_max_length(self) -> None:
        """Collate should pad sequences to max length in batch."""
        batch = [
            {
                'src_input_ids': [1, 10, 11, 2],
                'tgt_input_ids': [1, 20, 2],
                'src_attention_mask': [1, 1, 1, 1],
                'tgt_attention_mask': [1, 1, 1],
            },
            {
                'src_input_ids': [1, 10, 11, 12, 13, 2],
                'tgt_input_ids': [1, 20, 21, 22, 2],
                'src_attention_mask': [1, 1, 1, 1, 1, 1],
                'tgt_attention_mask': [1, 1, 1, 1, 1],
            },
        ]
        
        result = collate_fn(batch, pad_token_id=0)
        
        # All sequences should have same length
        assert result['src_input_ids'].shape[0] == 2
        assert result['tgt_input_ids'].shape[0] == 2
        
        # Should be padded to longest in batch
        assert result['src_input_ids'].shape[1] == 6
        assert result['tgt_input_ids'].shape[1] == 5

    def test_attention_mask_zeros_padding(self) -> None:
        """Attention mask should have 0s for padding positions."""
        batch = [
            {
                'src_input_ids': [1, 10, 2],
                'tgt_input_ids': [1, 20, 2],
                'src_attention_mask': [1, 1, 1],
                'tgt_attention_mask': [1, 1, 1],
            },
            {
                'src_input_ids': [1, 10, 11, 12, 2],
                'tgt_input_ids': [1, 20, 21, 2],
                'src_attention_mask': [1, 1, 1, 1, 1],
                'tgt_attention_mask': [1, 1, 1, 1],
            },
        ]
        
        result = collate_fn(batch, pad_token_id=0)
        
        # First sequence should have 0s at the end (padding)
        assert result['src_attention_mask'][0, -1].item() == 0
        assert result['src_attention_mask'][0, -2].item() == 0
        
        # Second sequence should have all 1s (no padding needed)
        assert result['src_attention_mask'][1].sum().item() == 5

    def test_returns_tensors(self) -> None:
        """Collate should return torch tensors."""
        batch = [
            {
                'src_input_ids': [1, 2, 3],
                'tgt_input_ids': [1, 2],
                'src_attention_mask': [1, 1, 1],
                'tgt_attention_mask': [1, 1],
            },
        ]
        
        result = collate_fn(batch, pad_token_id=0)
        
        assert isinstance(result['src_input_ids'], torch.Tensor)
        assert isinstance(result['tgt_input_ids'], torch.Tensor)
        assert isinstance(result['src_attention_mask'], torch.Tensor)
        assert isinstance(result['tgt_attention_mask'], torch.Tensor)


# ============================================================================
# TranslationTrainer Tests
# ============================================================================

class TestTranslationTrainer:
    """Tests for TranslationTrainer."""

    @pytest.fixture
    def tiny_model(self) -> MobileTranslationModel:
        """Create tiny model for testing."""
        return create_model(vocab_size=100, model_size='tiny')

    @pytest.fixture
    def mock_dataloader(self) -> DataLoader:
        """Create mock dataloader with a few batches."""
        data = [
            {
                'src_input_ids': torch.randint(0, 100, (2, 10)),
                'tgt_input_ids': torch.randint(0, 100, (2, 8)),
                'src_attention_mask': torch.ones(2, 10),
                'tgt_attention_mask': torch.ones(2, 8),
            }
            for _ in range(3)
        ]
        return data  # Return list, trainer iterates over it

    def test_trainer_initialization(self, tiny_model: MobileTranslationModel) -> None:
        """Trainer should initialize correctly."""
        mock_loader = [
            {
                'src_input_ids': torch.randint(0, 100, (2, 10)),
                'tgt_input_ids': torch.randint(0, 100, (2, 8)),
                'src_attention_mask': torch.ones(2, 10),
                'tgt_attention_mask': torch.ones(2, 8),
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TranslationTrainer(
                model=tiny_model,
                train_loader=mock_loader,
                device='cpu',
                save_dir=tmpdir,
                max_steps=100,
            )
            
            assert trainer.model is tiny_model
            assert trainer.global_step == 0

    def test_train_step_returns_loss(self, tiny_model: MobileTranslationModel) -> None:
        """Train step should return loss and metrics."""
        mock_loader = [
            {
                'src_input_ids': torch.randint(0, 100, (2, 10)),
                'tgt_input_ids': torch.randint(0, 100, (2, 8)),
                'src_attention_mask': torch.ones(2, 10),
                'tgt_attention_mask': torch.ones(2, 8),
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TranslationTrainer(
                model=tiny_model,
                train_loader=mock_loader,
                device='cpu',
                save_dir=tmpdir,
                max_steps=100,
            )
            
            batch = mock_loader[0]
            loss, metrics = trainer.train_step(batch)
            
            assert isinstance(loss, float)
            assert 'loss' in metrics
            assert 'grad_norm' in metrics

    def test_train_step_updates_parameters(self, tiny_model: MobileTranslationModel) -> None:
        """Train step should update model parameters."""
        mock_loader = [
            {
                'src_input_ids': torch.randint(0, 100, (2, 10)),
                'tgt_input_ids': torch.randint(0, 100, (2, 8)),
                'src_attention_mask': torch.ones(2, 10),
                'tgt_attention_mask': torch.ones(2, 8),
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TranslationTrainer(
                model=tiny_model,
                train_loader=mock_loader,
                device='cpu',
                save_dir=tmpdir,
                max_steps=100,
            )
            
            # Get initial parameters
            initial_params = {
                name: param.clone() for name, param in tiny_model.named_parameters()
            }
            
            # Run training step
            batch = mock_loader[0]
            trainer.train_step(batch)
            
            # Check that at least some parameters changed
            params_changed = False
            for name, param in tiny_model.named_parameters():
                if not torch.equal(initial_params[name], param):
                    params_changed = True
                    break
            
            assert params_changed, "Parameters should be updated after train step"

    def test_save_and_load_checkpoint(self, tiny_model: MobileTranslationModel) -> None:
        """Checkpoint save and load should work."""
        mock_loader = [
            {
                'src_input_ids': torch.randint(0, 100, (2, 10)),
                'tgt_input_ids': torch.randint(0, 100, (2, 8)),
                'src_attention_mask': torch.ones(2, 10),
                'tgt_attention_mask': torch.ones(2, 8),
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TranslationTrainer(
                model=tiny_model,
                train_loader=mock_loader,
                device='cpu',
                save_dir=tmpdir,
                max_steps=100,
            )
            
            # Run a step to change global_step
            trainer.train_step(mock_loader[0])
            step_before = trainer.global_step
            
            # Save checkpoint
            checkpoint_path = f"{tmpdir}/test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)
            
            assert Path(checkpoint_path).exists()
            
            # Reset step and load
            trainer.global_step = 0
            trainer.load_checkpoint(checkpoint_path)
            
            assert trainer.global_step == step_before

    def test_empty_batch_handling(self, tiny_model: MobileTranslationModel) -> None:
        """Train step should handle empty batches gracefully."""
        empty_batch = {
            'src_input_ids': torch.zeros(0, 10, dtype=torch.long),
            'tgt_input_ids': torch.zeros(0, 8, dtype=torch.long),
            'src_attention_mask': torch.zeros(0, 10),
            'tgt_attention_mask': torch.zeros(0, 8),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TranslationTrainer(
                model=tiny_model,
                train_loader=[empty_batch],
                device='cpu',
                save_dir=tmpdir,
                max_steps=100,
            )
            
            loss, metrics = trainer.train_step(empty_batch)
            
            # Should return 0 loss for empty batch
            assert loss == 0.0


# ============================================================================
# Loss Computation Tests  
# ============================================================================

class TestLossComputation:
    """Tests for loss computation in the model."""

    @pytest.fixture
    def tiny_model(self) -> MobileTranslationModel:
        """Create tiny model for testing."""
        return create_model(vocab_size=100, model_size='tiny')

    def test_compute_loss_returns_scalar(self, tiny_model: MobileTranslationModel) -> None:
        """compute_loss should return a scalar tensor."""
        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))
        
        loss = tiny_model.compute_loss(src, tgt)
        
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_loss_with_label_smoothing(self, tiny_model: MobileTranslationModel) -> None:
        """Label smoothing should affect loss value."""
        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))
        
        loss_no_smooth = tiny_model.compute_loss(src, tgt, label_smoothing=0.0)
        loss_smooth = tiny_model.compute_loss(src, tgt, label_smoothing=0.1)
        
        # With random weights, losses should be different
        # (though the relationship depends on the predictions)
        assert loss_no_smooth.item() != loss_smooth.item()

    def test_loss_ignores_padding(self, tiny_model: MobileTranslationModel) -> None:
        """Loss should ignore padding tokens."""
        src = torch.randint(0, 100, (2, 10))
        
        # Target with some padding
        tgt = torch.randint(1, 100, (2, 8))  # Avoid 0 (pad)
        tgt[0, 5:] = 0  # Add padding to first sequence
        
        # Should not raise, should handle padding
        loss = tiny_model.compute_loss(src, tgt)
        assert loss.item() > 0

    def test_loss_gradient_flow(self, tiny_model: MobileTranslationModel) -> None:
        """Loss should allow gradient flow."""
        src = torch.randint(0, 100, (2, 10))
        tgt = torch.randint(0, 100, (2, 8))
        
        loss = tiny_model.compute_loss(src, tgt)
        loss.backward()
        
        # Check that gradients exist
        has_grad = False
        for param in tiny_model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "Loss should produce gradients"
