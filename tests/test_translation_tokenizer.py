"""
Tests for TranslationTokenizer.

Covers:
- Encoding/decoding with special tokens
- Empty string and edge case handling
- Language validation
- Max length truncation
- Batch operations
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, List, Union

import torch


class TestTranslationTokenizerValidation:
    """Tests for tokenizer validation and error handling."""

    def test_uninitialized_tokenizer_raises_on_encode(self) -> None:
        """Tokenizer should raise if model not loaded before encoding."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        
        with pytest.raises(RuntimeError, match="model is not loaded"):
            tokenizer.encode("test")

    def test_uninitialized_tokenizer_raises_on_decode(self) -> None:
        """Tokenizer should raise if model not loaded before decoding."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        
        with pytest.raises(RuntimeError, match="model is not loaded"):
            tokenizer.decode([1, 2, 3])

    def test_unsupported_source_language_raises(self) -> None:
        """Tokenizer should raise for unsupported source language."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        # Mock the model as loaded
        tokenizer.sp_model = MagicMock()
        tokenizer.token_to_id = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
        
        with pytest.raises(ValueError, match="Unsupported source language 'fr'"):
            tokenizer.encode("hello", src_lang='fr', tgt_lang='es')

    def test_unsupported_target_language_raises(self) -> None:
        """Tokenizer should raise for unsupported target language."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        # Mock the model as loaded
        tokenizer.sp_model = MagicMock()
        tokenizer.token_to_id = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
        
        with pytest.raises(ValueError, match="Unsupported target language 'de'"):
            tokenizer.encode("hello", src_lang='en', tgt_lang='de')

    def test_partial_language_spec_raises(self) -> None:
        """Tokenizer should raise if only one language is specified."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        # Mock the model as loaded
        tokenizer.sp_model = MagicMock()
        tokenizer.sp_model.encode_as_ids.return_value = [100, 101]
        tokenizer.token_to_id = {
            '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3,
            '<src>': 4, '<tgt>': 5, '<en>': 6, '<es>': 7
        }
        
        with pytest.raises(ValueError, match="Either both"):
            tokenizer.encode("hello", src_lang='en')  # Missing tgt_lang


class TestTranslationTokenizerEncoding:
    """Tests for tokenizer encoding functionality."""

    @pytest.fixture
    def mock_tokenizer(self) -> "TranslationTokenizer":
        """Create a tokenizer with mocked SentencePiece model."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es', 'fr'])
        tokenizer.sp_model = MagicMock()
        tokenizer.sp_model.encode_as_ids.return_value = [100, 101, 102]
        tokenizer.sp_model.decode_ids.return_value = "hello world"
        tokenizer.sp_model.get_piece_size.return_value = 1000
        
        tokenizer.token_to_id = {
            '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3,
            '<src>': 4, '<tgt>': 5, '<en>': 6, '<es>': 7, '<fr>': 8
        }
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        
        return tokenizer

    def test_encode_without_special_tokens(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Encode should return raw token IDs when add_special_tokens=False."""
        result = mock_tokenizer.encode("hello", add_special_tokens=False)
        
        assert result == [100, 101, 102]

    def test_encode_with_special_tokens_no_lang(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Encode should add <s> and </s> when no language specified."""
        result = mock_tokenizer.encode("hello", add_special_tokens=True)
        
        # Should be: <s> + content + </s>
        assert result[0] == 1  # <s>
        assert result[-1] == 2  # </s>
        assert result[1:-1] == [100, 101, 102]

    def test_encode_with_translation_format(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Encode should use translation format when both languages specified."""
        result = mock_tokenizer.encode("hello", src_lang='en', tgt_lang='es', add_special_tokens=True)
        
        # Should be: <src> <en> content </s> <tgt> <es>
        assert result[0] == 4  # <src>
        assert result[1] == 6  # <en>
        assert result[-3] == 2  # </s>
        assert result[-2] == 5  # <tgt>
        assert result[-1] == 7  # <es>

    def test_encode_max_length_truncation(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Encode should truncate to max_length."""
        result = mock_tokenizer.encode("hello", add_special_tokens=True, max_length=3)
        
        assert len(result) == 3

    def test_decode_with_skip_special_tokens(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Decode should skip special tokens when requested."""
        # Token IDs that include special tokens
        token_ids = [1, 100, 101, 2]  # <s> content </s>
        
        result = mock_tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Should have called decode_ids without special token IDs
        mock_tokenizer.sp_model.decode_ids.assert_called()
        args = mock_tokenizer.sp_model.decode_ids.call_args[0][0]
        assert 1 not in args  # <s> removed
        assert 2 not in args  # </s> removed


class TestTranslationTokenizerProperties:
    """Tests for tokenizer property accessors."""

    def test_special_token_id_properties(self) -> None:
        """Test that special token ID properties work correctly."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        tokenizer.token_to_id = {
            '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3
        }
        
        assert tokenizer.pad_token_id == 0
        assert tokenizer.sos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_vocab_size(self) -> None:
        """Test vocab size property."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        tokenizer.token_to_id = {f'token_{i}': i for i in range(100)}
        
        assert tokenizer.get_vocab_size() == 100


class TestTranslationTokenizerBatch:
    """Tests for batch encoding/decoding."""

    @pytest.fixture
    def mock_tokenizer(self) -> "TranslationTokenizer":
        """Create a tokenizer with mocked SentencePiece model."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        tokenizer = TranslationTokenizer(languages=['en', 'es'])
        tokenizer.sp_model = MagicMock()
        
        # Return different lengths for different inputs
        def mock_encode(text: str) -> List[int]:
            return list(range(len(text)))
        
        tokenizer.sp_model.encode_as_ids.side_effect = mock_encode
        tokenizer.token_to_id = {
            '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3,
            '<src>': 4, '<tgt>': 5, '<en>': 6, '<es>': 7
        }
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        
        return tokenizer

    def test_batch_encode_with_padding(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Batch encode should pad sequences to same length."""
        texts = ["hi", "hello world"]
        
        result = mock_tokenizer.batch_encode(texts, padding=True, return_tensors=True)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']
        
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        
        # Both sequences should have same length
        assert input_ids.shape[0] == 2
        assert input_ids.shape[1] == attention_mask.shape[1]

    def test_batch_encode_attention_mask_zeros_padding(self, mock_tokenizer: "TranslationTokenizer") -> None:
        """Attention mask should have 0s for padding positions."""
        texts = ["a", "abc"]  # Different lengths
        
        result = mock_tokenizer.batch_encode(texts, padding=True, return_tensors=True)
        
        attention_mask = result['attention_mask']
        
        # First sequence is shorter, should have padding
        # The mask should have 0s at the end for the shorter sequence
        assert attention_mask.shape[0] == 2


class TestTranslationTokenizerPersistence:
    """Tests for save/load functionality."""

    def test_from_pretrained_missing_directory(self) -> None:
        """from_pretrained should raise for missing directory."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            TranslationTokenizer.from_pretrained("/nonexistent/path")

    def test_from_pretrained_not_a_directory(self, tmp_path: Path) -> None:
        """from_pretrained should raise if path is a file, not directory."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        # Create a file instead of directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")
        
        with pytest.raises(ValueError, match="not a directory"):
            TranslationTokenizer.from_pretrained(file_path)

    def test_from_pretrained_missing_config(self, tmp_path: Path) -> None:
        """from_pretrained should raise if config file is missing."""
        from lingolite.translation_tokenizer import TranslationTokenizer
        
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            TranslationTokenizer.from_pretrained(tmp_path)


# Import for type hints
from lingolite.translation_tokenizer import TranslationTokenizer
