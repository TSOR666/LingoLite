"""
Multilingual Translation Tokenizer
Specialized tokenizer for limited language pairs with efficient vocabulary
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import sentencepiece as spm  # type: ignore[import-untyped]
import torch


class TranslationTokenizer:
    """
    Tokenizer optimized for translation between specific language pairs.
    Uses SentencePiece (Unigram) for subword tokenization.
    """
    
    def __init__(
        self,
        languages: List[str] = ['en', 'es', 'fr', 'de', 'it', 'da'],
        vocab_size: int = 24000,
        model_prefix: str = "translation_tokenizer"
    ):
        """
        Args:
            languages: List of language codes to support
            vocab_size: Size of vocabulary (smaller than general LLMs)
            model_prefix: Prefix for saved model files
        """
        self.languages = languages
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        
        # Language tokens
        self.lang_tokens = [f"<{lang}>" for lang in languages]
        
        # Special markers
        self.src_token = "<src>"
        self.tgt_token = "<tgt>"
        
        # All special tokens
        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
            self.src_token,
            self.tgt_token,
        ] + self.lang_tokens
        
        self.sp_model: Optional[spm.SentencePieceProcessor] = None
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def _ensure_model_loaded(self) -> None:
        """Ensure the SentencePiece model has been loaded before encoding/decoding."""
        if self.sp_model is None or not self.token_to_id:
            raise RuntimeError(
                "Tokenizer model is not loaded. Call 'train' or 'load' before encoding/decoding."
            )
        assert self.sp_model is not None

    def _validate_language(self, lang: str, role: str) -> None:
        """Validate that a requested language is supported."""
        if lang not in self.languages:
            raise ValueError(
                f"Unsupported {role} language '{lang}'. Supported languages: {self.languages}"
            )
        
    def train(
        self,
        corpus_files: List[str],
        character_coverage: float = 0.9995,
        model_type: str = "unigram"
    ) -> None:
        """
        Train tokenizer on multilingual corpus.
        
        Args:
            corpus_files: List of text files for training
            character_coverage: Coverage of characters (0.9995 for most languages)
            model_type: 'unigram' or 'bpe'
        """
        print(f"Training tokenizer on {len(corpus_files)} files...")
        
        # SentencePiece training arguments
        spm.SentencePieceTrainer.train(
            input=','.join(corpus_files),
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            unk_id=3,
            bos_id=1,
            eos_id=2,
            pad_piece=self.pad_token,
            unk_piece=self.unk_token,
            bos_piece=self.sos_token,
            eos_piece=self.eos_token,
            user_defined_symbols=self.lang_tokens + [self.src_token, self.tgt_token],
            num_threads=8,
        )
        
        # Load trained model
        self.load(f"{self.model_prefix}.model")
        
        print(f"âœ“ Tokenizer trained with vocab size: {len(self.token_to_id)}")
        
    def load(self, model_path: Union[str, Path]) -> None:
        """Load trained tokenizer model."""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(model_path))
        
        # Build token mappings
        self.token_to_id = {
            self.sp_model.id_to_piece(i): i 
            for i in range(self.sp_model.get_piece_size())
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        print(f"âœ“ Tokenizer loaded from {model_path}")
        
    def save(self, save_dir: Union[str, Path]) -> None:
        """Save tokenizer and config."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'languages': self.languages,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'model_prefix': self.model_prefix,
        }
        
        with open(save_dir / 'tokenizer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        # Copy SentencePiece files
        import shutil
        from pathlib import Path as _Path
        for ext in ('.model', '.vocab'):
            src = _Path(f"{self.model_prefix}{ext}")
            if src.exists():
                shutil.copyfile(src, save_dir / src.name)
        print(f"âœ“ Tokenizer saved to {save_dir} (config + SentencePiece files)")
        
    @classmethod
    def from_pretrained(cls, load_dir: Union[str, Path]) -> "TranslationTokenizer":
        """Load tokenizer from directory."""
        # SECURITY: Validate and resolve path
        load_dir = Path(load_dir).resolve()
        if not load_dir.exists():
            raise FileNotFoundError(f"Directory not found: {load_dir}")
        if not load_dir.is_dir():
            raise ValueError(f"Path is not a directory: {load_dir}")

        # Load config
        config_path = load_dir / 'tokenizer_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        tokenizer = cls(
            languages=config['languages'],
            vocab_size=config['vocab_size'],
            model_prefix=config.get('model_prefix', 'translation_tokenizer'),
        )

        # Load model
        model_path = load_dir / f"{tokenizer.model_prefix}.model"
        tokenizer.load(str(model_path))

        return tokenizer
    
    def encode(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            src_lang: Source language code (for translation format)
            tgt_lang: Target language code (for translation format)
            add_special_tokens: Whether to add <s>, </s>, language tokens
            max_length: Maximum sequence length (truncate if exceeded)
        
        Returns:
            List of token IDs
        """
        self._ensure_model_loaded()
        assert self.sp_model is not None
        sp_model = cast(spm.SentencePieceProcessor, self.sp_model)

        # Encode text
        token_ids = cast(List[int], sp_model.encode_as_ids(text))

        if add_special_tokens:
            tokens = []

            if bool(src_lang) ^ bool(tgt_lang):
                raise ValueError("Either both 'src_lang' and 'tgt_lang' must be provided together, or neither should be provided.")

            # Translation format: <src> <lang> text </s> <tgt> <lang>
            if src_lang and tgt_lang:
                self._validate_language(src_lang, 'source')
                self._validate_language(tgt_lang, 'target')
                tokens.extend([
                    self.token_to_id[self.src_token],
                    self.token_to_id[f"<{src_lang}>"]
                ])
                tokens.extend(token_ids)
                tokens.extend([
                    self.token_to_id[self.eos_token],
                    self.token_to_id[self.tgt_token],
                    self.token_to_id[f"<{tgt_lang}>"]
                ])
            else:
                # Standard format: <s> text </s>
                tokens = [self.token_to_id[self.sos_token]] + token_ids + [self.token_to_id[self.eos_token]]
            
            token_ids = tokens
        
        # Truncate if needed
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens
        
        Returns:
            Decoded text
        """
        self._ensure_model_loaded()
        assert self.sp_model is not None
        sp_model = cast(spm.SentencePieceProcessor, self.sp_model)

        if skip_special_tokens:
            # Remove special tokens
            special_ids = {self.token_to_id.get(tok, -1) for tok in self.special_tokens}
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        text = cast(str, sp_model.decode_ids(token_ids))
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: bool = True
    ) -> Dict[str, Union[torch.Tensor, List[List[int]]]]:
        """
        Batch encode texts with padding.
        
        Args:
            texts: List of texts
            src_lang: Source language
            tgt_lang: Target language
            padding: Whether to pad to same length
            max_length: Maximum sequence length
            return_tensors: Whether to return PyTorch tensors
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if src_lang:
            self._validate_language(src_lang, 'source')
        if tgt_lang:
            self._validate_language(tgt_lang, 'target')

        # Encode all texts
        encoded = [
            self.encode(text, src_lang, tgt_lang, max_length=max_length)
            for text in texts
        ]
        
        if padding:
            # Find max length
            max_len = max(len(ids) for ids in encoded)
            if max_length:
                max_len = min(max_len, max_length)
            
            # Pad sequences
            pad_id = self.token_to_id[self.pad_token]
            padded = []
            attention_masks = []
            
            for ids in encoded:
                # Truncate if needed
                if len(ids) > max_len:
                    ids = ids[:max_len]
                
                # Create attention mask (1 for real tokens, 0 for padding)
                mask = [1] * len(ids) + [0] * (max_len - len(ids))
                
                # Pad
                ids = ids + [pad_id] * (max_len - len(ids))
                
                padded.append(ids)
                attention_masks.append(mask)
            
            encoded = padded
            
            if return_tensors:
                result: Dict[str, Union[torch.Tensor, List[List[int]]]] = {
                    'input_ids': torch.tensor(encoded, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
                }
                return result
            else:
                result = {
                    'input_ids': encoded,
                    'attention_mask': attention_masks
                }
                return result
        
        if return_tensors:
            return {'input_ids': torch.tensor(encoded, dtype=torch.long)}
        return {'input_ids': encoded}
    
    def batch_decode(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Batch decode token IDs to texts.
        
        Args:
            token_ids_batch: List of token ID sequences
            skip_special_tokens: Whether to remove special tokens
        
        Returns:
            List of decoded texts
        """
        return [
            self.decode(token_ids, skip_special_tokens)
            for token_ids in token_ids_batch
        ]
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]
    
    @property
    def sos_token_id(self) -> int:
        return self.token_to_id[self.sos_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]
    
    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.unk_token]
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)


# Example usage and testing
if __name__ == "__main__":
    pass
