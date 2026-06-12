"""
Training Script for Mobile Translation Model
Complete pipeline from data to trained model
"""

import argparse
import functools
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # type: ignore[import-untyped]

from .mobile_translation_model import MobileTranslationModel, create_model
from .translation_tokenizer import TranslationTokenizer
from .utils import setup_logger

# Set up logger
logger = setup_logger(name="lingolite_training", level=logging.INFO)


class TranslationDataset(Dataset[Dict[str, List[int]]]):
    """
    Dataset for translation pairs.
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
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
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
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

        # Tokenize target (format: <s> text </s>). The source sequence already
        # carries the requested target language as <tgt> <lang>, and generation
        # starts from sos_token_id, so training must use the same decoder start.
        tgt_text = item['tgt_text']
        target_body_max_length = max(0, self.max_length - 2)
        tgt_ids = self.tokenizer.encode(
            text=tgt_text,
            add_special_tokens=False,
            max_length=target_body_max_length,
        )
        
        # Add SOS and EOS while respecting max_length.
        if item['tgt_lang'] not in self.tokenizer.languages:
            raise ValueError(f"Target language '{item['tgt_lang']}' not supported. Supported: {self.tokenizer.languages}")
        sos_token_id = getattr(self.tokenizer, 'sos_token_id', self.tokenizer.token_to_id.get('<s>'))
        if sos_token_id is None:
            raise ValueError("Tokenizer must expose sos_token_id or token_to_id['<s>']")
        tgt_ids = [int(sos_token_id)] + tgt_ids + [self.tokenizer.eos_token_id]
        
        src_mask = [1] * len(src_ids)
        tgt_mask = [1] * len(tgt_ids)
        
        return {
            'src_input_ids': src_ids,
            'tgt_input_ids': tgt_ids,
            'src_attention_mask': src_mask,
            'tgt_attention_mask': tgt_mask,
        }


def collate_fn(batch: List[Dict[str, List[int]]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
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
    
    # Support both dataset item formats:
    # - legacy keys: src_ids/tgt_ids
    # - standardized keys: src_input_ids/tgt_input_ids (+ optional masks)
    use_standard_keys = (
        'src_input_ids' in batch[0]
        and 'tgt_input_ids' in batch[0]
    )

    src_key = 'src_input_ids' if use_standard_keys else 'src_ids'
    tgt_key = 'tgt_input_ids' if use_standard_keys else 'tgt_ids'
    src_mask_key = 'src_attention_mask' if use_standard_keys else None
    tgt_mask_key = 'tgt_attention_mask' if use_standard_keys else None

    # Find max lengths
    def _get(key: str, item: Dict[str, List[int]]) -> List[int]:
        # Support both legacy keys ('src_ids') and new keys ('src_input_ids')
        if key in item:
            return item[key]
        legacy_key = key.replace("_input", "")
        return item[legacy_key]
    
    max_src_len = max(len(_get('src_input_ids', item)) for item in batch)
    max_tgt_len = max(len(_get('tgt_input_ids', item)) for item in batch)
    
    # Pad sequences
    src_ids = []
    tgt_ids = []
    src_mask = []
    tgt_mask = []
    
    for item in batch:
        # Pad source
        src = _get('src_input_ids', item)
        src_padding = [pad_token_id] * (max_src_len - len(src))
        src_ids.append(src + src_padding)

        item_src_mask = item.get(src_mask_key) if src_mask_key is not None else None
        if item_src_mask is None:
            src_mask_values = [1] * len(src)
        else:
            # Clamp to source length and force binary mask semantics.
            src_mask_values = [1 if int(v) != 0 else 0 for v in item_src_mask[:len(src)]]
            if len(src_mask_values) < len(src):
                src_mask_values += [1] * (len(src) - len(src_mask_values))
        src_mask.append(src_mask_values + [0] * len(src_padding))
        
        # Pad target
        tgt = _get('tgt_input_ids', item)
        tgt_padding = [pad_token_id] * (max_tgt_len - len(tgt))
        tgt_ids.append(tgt + tgt_padding)

        item_tgt_mask = item.get(tgt_mask_key) if tgt_mask_key is not None else None
        if item_tgt_mask is None:
            tgt_mask_values = [1] * len(tgt)
        else:
            tgt_mask_values = [1 if int(v) != 0 else 0 for v in item_tgt_mask[:len(tgt)]]
            if len(tgt_mask_values) < len(tgt):
                tgt_mask_values += [1] * (len(tgt) - len(tgt_mask_values))
        tgt_mask.append(tgt_mask_values + [0] * len(tgt_padding))
    
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
        train_loader: DataLoader[Dict[str, torch.Tensor]],
        val_loader: Optional[DataLoader[Dict[str, torch.Tensor]]] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        gradient_clip: float = 1.0,
        label_smoothing: float = 0.1,
        device: str = 'cuda',
        save_dir: str = './checkpoints',
        amp_dtype: Optional[str] = None,
        compile_model: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
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
            amp_dtype: Optional autocast dtype ("fp16", "bf16", or None).
                On CUDA, fp16 enables GradScaler-backed mixed precision for
                ~2x throughput; bf16 skips GradScaler (no under/overflow) and
                needs an Ampere+ GPU. On CPU, bf16 is the only viable choice.
            compile_model: When True, wrap the model in ``torch.compile`` for
                kernel fusion. Falls back gracefully if compile is unavailable.
            gradient_accumulation_steps: Average gradients over this many
                micro-batches before stepping the optimizer. Effective batch
                size becomes ``train_batch_size * gradient_accumulation_steps``
                without extra peak memory. ``global_step`` advances per
                optimizer step, not per micro-batch, so eval/save cadence is
                unchanged. Defaults to 1 (no accumulation).
            gradient_checkpointing: When True, recompute encoder/decoder layer
                activations during backward instead of caching them. Trades
                ~30% extra compute for substantially lower training-time
                activation memory. Only active when ``model.training`` is True
                so inference paths (greedy/beam) are unaffected.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Parse AMP configuration.
        amp_dtype_map: Dict[str, torch.dtype] = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        self.amp_dtype: Optional[torch.dtype] = None
        if amp_dtype is not None:
            key = amp_dtype.lower()
            if key not in amp_dtype_map:
                raise ValueError(
                    f"amp_dtype must be one of {list(amp_dtype_map)} or None, got {amp_dtype!r}"
                )
            self.amp_dtype = amp_dtype_map[key]
            if self.amp_dtype is torch.float16 and device != "cuda":
                raise ValueError("amp_dtype='fp16' requires CUDA; use 'bf16' on CPU.")

        # GradScaler is only needed for fp16 (bf16 has the same dynamic range
        # as fp32 and doesn't under/overflow).
        self._amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
        self._use_grad_scaler = self.amp_dtype is torch.float16
        if self._use_grad_scaler:
            self.scaler = torch.amp.GradScaler(self._amp_device)
        else:
            self.scaler = None  # type: ignore[assignment]

        # Optional torch.compile wrapping (opt-in; not all torch builds ship
        # with Inductor). We intentionally compile the module and let
        # compute_loss dispatch through it.
        if compile_model:
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception as exc:
                logger.warning("torch.compile unavailable, continuing uncompiled: %s", exc)

        # Validate max_steps
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        self.max_steps = max_steps

        # Build weight-decay parameter groups: exclude normalisation weights and
        # embeddings from L2 decay (standard transformer training practice).
        # Biases are already disabled in this architecture, so we only have to
        # filter on parameter names.
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        seen_param_ids: set[int] = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Skip duplicates introduced by weight tying (encoder.embedding is
            # aliased to decoder.embedding and to decoder.lm_head).
            pid = id(param)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            lowered = name.lower()
            if "norm" in lowered or "embedding" in lowered or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Clamp warmup to keep OneCycleLR pct_start in [0, 1).
        max_warmup_steps = max(0, max_steps - 1)
        self.warmup_steps = min(warmup_steps, max_warmup_steps)
        if self.warmup_steps != warmup_steps:
            logger.warning(
                "warmup_steps (%d) exceeds max_steps-1 (%d); clamping to %d",
                warmup_steps,
                max_warmup_steps,
                self.warmup_steps,
            )

        # OneCycleLR with a very short ``total_steps`` and a large
        # ``final_div_factor`` collapses the LR to ~1e-9 within a handful of
        # steps (e.g. smoke tests). Fall back to a linear-warmup + cosine-decay
        # schedule when ``max_steps`` is too small for OneCycleLR to behave
        # reasonably (heuristic: need at least ~20 post-warmup decay steps).
        post_warmup_steps = max_steps - self.warmup_steps
        if post_warmup_steps < 20:
            warmup = max(self.warmup_steps, 1)
            total = max(max_steps, 1)

            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step + 1) / float(warmup)
                progress = float(step - warmup) / float(max(1, total - warmup))
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            logger.info(
                "Using LinearWarmup+CosineDecay fallback (max_steps=%d too small "
                "for OneCycleLR; need >=20 post-warmup steps, got %d)",
                max_steps,
                post_warmup_steps,
            )
        else:
            pct_start = min(max(float(self.warmup_steps) / float(max_steps), 0.0), 1.0)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=max_steps,
                pct_start=pct_start,
                anneal_strategy='cos',
            )

        self.gradient_clip = gradient_clip
        self.label_smoothing = label_smoothing

        if gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"
            )
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        # Counts micro-batches consumed within the current accumulation cycle.
        # An optimizer step (and global_step increment) fires when this hits
        # ``gradient_accumulation_steps``.
        self._micro_step_in_cycle = 0

        # Optionally enable activation checkpointing on the wrapped model.
        # The encoder/decoder check ``self.training`` internally, so this is a
        # no-op for eval / generate paths even when the flag is on.
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            enable = getattr(self.model, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()
            else:
                logger.warning(
                    "gradient_checkpointing=True but model has no "
                    "gradient_checkpointing_enable(); ignoring."
                )

        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Single training step.

        With ``gradient_accumulation_steps > 1`` this runs a *micro-step*: the
        loss is scaled by ``1/N``, ``backward()`` accumulates gradients into
        the existing buffers, and the optimizer/scheduler/scaler only fire on
        the Nth micro-step. ``global_step`` advances per optimizer step, so
        callers reading ``self.global_step`` see the same cadence as before
        regardless of the accumulation factor.

        Args:
            batch: Batch of data

        Returns:
            loss: Per-micro-batch loss (unscaled, for logging)
            metrics: Dictionary of metrics
        """
        # Handle empty batch
        if batch['src_input_ids'].numel() == 0:
            return 0.0, {'loss': 0.0, 'grad_norm': 0.0, 'lr': 0.0}

        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        accum = self.gradient_accumulation_steps
        is_first_micro = self._micro_step_in_cycle == 0
        is_last_micro = self._micro_step_in_cycle == accum - 1

        # Only zero gradients at the start of a new accumulation cycle so
        # intermediate micro-steps add to the running gradient buffers.
        if is_first_micro:
            self.optimizer.zero_grad()

        # Forward pass (optionally under autocast for mixed precision).
        if self.amp_dtype is not None:
            with torch.autocast(device_type=self._amp_device, dtype=self.amp_dtype):
                loss: torch.Tensor = self.model.compute_loss(
                    src_input_ids=batch['src_input_ids'],
                    tgt_input_ids=batch['tgt_input_ids'],
                    src_attention_mask=batch['src_attention_mask'],
                    tgt_attention_mask=batch['tgt_attention_mask'],
                    label_smoothing=self.label_smoothing,
                )
        else:
            loss = self.model.compute_loss(
                src_input_ids=batch['src_input_ids'],
                tgt_input_ids=batch['tgt_input_ids'],
                src_attention_mask=batch['src_attention_mask'],
                tgt_attention_mask=batch['tgt_attention_mask'],
                label_smoothing=self.label_smoothing,
            )

        # Scale the loss so summed gradients across the cycle equal the mean
        # gradient of the equivalent large batch. Skip the divide when accum=1
        # to avoid an unnecessary op on the hot path.
        loss_for_backward = loss if accum == 1 else loss / accum

        # Backward pass (scaled for fp16 to avoid underflow).
        # ``grad_norm`` stays a plain float on mid-cycle micro-steps so the
        # metrics dict below costs no extra device->host sync for them.
        grad_norm_value = 0.0
        if self._use_grad_scaler and self.scaler is not None:
            self.scaler.scale(loss_for_backward).backward()
            if is_last_micro:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip,
                    norm_type=2.0,
                )
                grad_norm_value = float(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss_for_backward.backward()  # type: ignore[no-untyped-call]
            if is_last_micro:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip,
                    norm_type=2.0,
                )
                grad_norm_value = float(grad_norm)
                self.optimizer.step()

        if is_last_micro:
            self.scheduler.step()
            self.global_step += 1
            self._micro_step_in_cycle = 0
        else:
            self._micro_step_in_cycle += 1

        # Sync the loss to host exactly once (it was previously read twice).
        loss_value = loss.item()
        metrics = {
            'loss': loss_value,
            'grad_norm': grad_norm_value,
            'lr': self.scheduler.get_last_lr()[0],
        }

        return loss_value, metrics
    
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

        # Accumulate on-device so the eval loop doesn't pay a host sync per
        # batch. Previously each iteration called ``.item()`` twice (loss and
        # token count), each one a CUDA->CPU sync that stalled the queue. We
        # now keep running totals in fp64 device tensors and only sync once
        # at the very end of the loop.
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        total_tokens = torch.zeros((), device=self.device, dtype=torch.float64)

        with torch.inference_mode():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Skip empty batches (numel is a tensor-metadata read, no sync).
                if batch['src_input_ids'].numel() == 0:
                    continue

                batch = {k: v.to(self.device) for k, v in batch.items()}

                loss = self.model.compute_loss(
                    src_input_ids=batch['src_input_ids'],
                    tgt_input_ids=batch['tgt_input_ids'],
                    src_attention_mask=batch['src_attention_mask'],
                    tgt_attention_mask=batch['tgt_attention_mask'],
                    label_smoothing=0.0,  # No smoothing for validation
                )

                # ``compute_loss`` returns the mean over non-pad target tokens,
                # so weighting by num_tokens recovers the per-batch sum.
                num_tokens = batch['tgt_attention_mask'][:, 1:].sum()
                total_loss = total_loss + loss.detach().to(torch.float64) * num_tokens.to(torch.float64)
                total_tokens = total_tokens + num_tokens.to(torch.float64)

        # Single host sync at the end.
        total_loss_value = float(total_loss.item())
        total_tokens_value = float(total_tokens.item())

        avg_loss = total_loss_value / max(1e-8, total_tokens_value)
        perplexity = float(np.exp(avg_loss))

        return {
            'val_loss': avg_loss,
            'perplexity': perplexity,
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint with RNG state for deterministic resume."""
        # Drop non-serializable entries from the scheduler state. On torch < 2.4
        # OneCycleLR.state_dict() contains ``anneal_func`` (a bound method),
        # which torch.load(..., weights_only=True) refuses to unpickle —
        # breaking every checkpoint reload. The fresh scheduler instance
        # already has the callable, so omitting it from the saved state is
        # lossless (LRScheduler.load_state_dict only updates the keys present).
        scheduler_state = {
            k: v for k, v in self.scheduler.state_dict().items() if not callable(v)
        }
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            # RNG state tensors are plain ByteTensors and are compatible with
            # torch.load(..., weights_only=True).
            'torch_rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint['torch_cuda_rng_state'] = torch.cuda.get_rng_state_all()

        checkpoint_path = Path(filename)
        save_path = checkpoint_path if checkpoint_path.is_absolute() else self.save_dir / checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"[OK] Checkpoint saved: {save_path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(filename)
        load_path = checkpoint_path if checkpoint_path.is_absolute() else self.save_dir / checkpoint_path
        # SECURITY: Use weights_only=True to prevent arbitrary code execution
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        # Restore RNG state if the checkpoint contains it (older checkpoints omit).
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'].to('cpu'))
        cuda_rng = checkpoint.get('torch_cuda_rng_state')
        if cuda_rng is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_rng)
            except Exception as exc:  # mismatched device count etc.
                logger.warning("Could not restore CUDA RNG state: %s", exc)

        print(f"[OK] Checkpoint loaded: {load_path}")
    
    def train(
        self,
        num_epochs: int,
        eval_steps: int = 5000,
        save_steps: int = 10000,
        log_steps: int = 100,
    ) -> None:
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

        running_loss: float = 0.0
        running_loss_count = 0
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
                    print(f"\n[OK] Reached max_steps ({self.max_steps})")
                    training_complete = True
                    break

                previous_global_step = self.global_step
                loss, metrics = self.train_step(batch)
                running_loss += loss
                running_loss_count += 1
                did_optimizer_step = self.global_step > previous_global_step

                # Update progress bar
                if did_optimizer_step and self.global_step % log_steps == 0:
                    avg_loss = running_loss / max(1, running_loss_count)
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{metrics['lr']:.2e}",
                        'step': self.global_step,
                    })
                    running_loss = 0
                    running_loss_count = 0

                # Evaluation
                if (
                    did_optimizer_step
                    and self.global_step % eval_steps == 0
                    and self.val_loader is not None
                ):
                    val_metrics = self.evaluate()
                    print(f"\nStep {self.global_step} - Validation:")
                    print(f"  Loss: {val_metrics['val_loss']:.4f}")
                    print(f"  Perplexity: {val_metrics['perplexity']:.2f}")

                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best_model.pt')

                # Save checkpoint
                if did_optimizer_step and self.global_step % save_steps == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        print(f"\n{'='*80}")
        print("[OK] TRAINING COMPLETE")
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
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes for train/validation loading')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Enable DataLoader pin_memory for faster CPU-to-CUDA transfers')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='Keep DataLoader workers alive between epochs (requires --num-workers > 0)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='DataLoader prefetch factor when --num-workers > 0')
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

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Mixed precision & compile
    parser.add_argument('--amp-dtype', type=str, default=None,
                        choices=['fp16', 'bf16'],
                        help='Enable autocast mixed precision. fp16 requires CUDA; bf16 works on Ampere+ CUDA or CPU')
    parser.add_argument('--compile', dest='compile_model', action='store_true',
                        help='Wrap the model in torch.compile for kernel fusion')

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line interface for model training."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Set random seed for reproducibility
    from .utils import set_seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed} for reproducibility")

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

    if args.num_workers < 0:
        logger.error(f"num_workers must be non-negative, got {args.num_workers}")
        sys.exit(1)
    if args.prefetch_factor <= 0:
        logger.error(f"prefetch_factor must be positive, got {args.prefetch_factor}")
        sys.exit(1)
    if args.persistent_workers and args.num_workers == 0:
        logger.error("--persistent-workers requires --num-workers > 0")
        sys.exit(1)

    dataloader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': bool(args.pin_memory),
    }
    if args.num_workers > 0:
        dataloader_kwargs['persistent_workers'] = bool(args.persistent_workers)
        dataloader_kwargs['prefetch_factor'] = args.prefetch_factor

    # Load validation data
    val_loader = None
    if args.val_data:
        print(f"\nLoading validation data from {args.val_data}...")
        with open(args.val_data, 'r') as f:
            val_data = json.load(f)
        print(f"[OK] Loaded {len(val_data)} validation examples")

        val_dataset = TranslationDataset(val_data, tokenizer, max_length=args.max_length)
        # functools.partial of a module-level function is picklable; a lambda is
        # not, and crashes DataLoader workers on spawn-based platforms (Windows,
        # macOS) whenever --num-workers > 0.
        val_loader = cast(
            DataLoader[Dict[str, torch.Tensor]],
            DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
                **dataloader_kwargs,
            ),
        )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TranslationDataset(train_data, tokenizer, max_length=args.max_length)
    train_loader = cast(
        DataLoader[Dict[str, torch.Tensor]],
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
            **dataloader_kwargs,
        ),
    )
    print(f"[OK] Train loader: {len(train_loader)} batches")
    if val_loader:
        print(f"[OK] Val loader: {len(val_loader)} batches")

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
    print(f"[OK] Model created: {params['total']/1e6:.1f}M parameters")

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
        amp_dtype=args.amp_dtype,
        compile_model=args.compile_model,
    )
    print("[OK] Trainer ready")

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
        print("[OK] TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("[WARN] TRAINING INTERRUPTED")
        print("=" * 80)
        print("Saving checkpoint...")
        trainer.save_checkpoint('interrupted.pt')
        print("[OK] Checkpoint saved")
        sys.exit(1)
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("[ERROR] TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
