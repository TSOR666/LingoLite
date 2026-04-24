"""End-to-end CPU smoke training run for LingoLite.

Trains a tiny model on the toy translation dataset until it overfits and then
saves a checkpoint + the smoke tokenizer state so that ``scripts/smoke_infer``
can load the artifacts and decode translations.

Success criteria enforced by this script:

1. The training loop actually steps (``global_step > 0``).
2. The loss decreases across steps (we require at least a 50% drop from the
   first logged step to the final step).
3. The final loss drops below a fixed convergence threshold.
4. A checkpoint file and a tokenizer JSON file are written to disk.

If any of these fails the script exits non-zero so CI / pytest can catch
regressions in the full train pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationDataset, TranslationTrainer, collate_fn
from scripts.smoke_common import (
    DEFAULT_TRAIN_PATH,
    PAD_ID,
    SmokeTokenizer,
    load_pairs,
)


@dataclass
class SmokeTrainResult:
    trainer: TranslationTrainer
    tokenizer: SmokeTokenizer
    losses: List[float]
    checkpoint_path: Path
    tokenizer_path: Path


def run_smoke_train(
    data_path: Path = DEFAULT_TRAIN_PATH,
    out_dir: Path = Path(".tmp_manual/smoke_train"),
    max_steps: int = 400,
    batch_size: int = 4,
    seed: int = 0,
    convergence_loss: float = 0.15,
    monotonic_ratio: float = 0.15,
) -> SmokeTrainResult:
    """Train a tiny model on the toy dataset and save a checkpoint.

    Args:
        data_path: JSON file with translation pairs.
        out_dir: Directory where the checkpoint + tokenizer state are written.
        max_steps: Hard cap on training iterations. Kept small enough that the
            whole run completes in seconds on CPU.
        batch_size: Batch size for the overfit loop.
        seed: Torch seed for reproducibility.
        convergence_loss: Final-loss threshold the overfit run must clear.
        monotonic_ratio: The last logged loss must be below
            ``monotonic_ratio * first_loss``. Guards against "training loop
            runs but doesn't learn" regressions.
    """
    torch.manual_seed(seed)

    pairs = load_pairs(data_path)
    tokenizer = SmokeTokenizer.from_pairs(pairs)

    dataset = TranslationDataset(pairs, tokenizer, max_length=24)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=PAD_ID),
    )

    vocab_size = tokenizer.get_vocab_size()
    model = create_model(
        vocab_size=vocab_size,
        model_size="tiny",
        d_model=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
        pad_token_id=PAD_ID,
    )
    print(f"Model vocab_size={vocab_size}, total_params={sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    # For the overfit smoke run we want the LR to peak quickly and *stay*
    # close to the peak for most of the run -- the default OneCycleLR cosine
    # decay otherwise collapses LR before the model has finished memorizing
    # the ~40 pairs. ``warmup_steps = max_steps // 3`` keeps ~2/3 of the run
    # in the high-LR phase.
    warmup_steps = max(20, max_steps // 3)
    trainer = TranslationTrainer(
        model=model,
        train_loader=loader,
        learning_rate=5e-3,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_clip=1.0,
        label_smoothing=0.0,
        device="cpu",
        save_dir=str(out_dir),
    )

    losses: List[float] = []
    first_logged_loss: Optional[float] = None
    for epoch in range(1, max_steps + 1):
        if trainer.global_step >= trainer.max_steps:
            break
        for batch in loader:
            if trainer.global_step >= trainer.max_steps:
                break
            loss, metrics = trainer.train_step(batch)
            if not torch.isfinite(torch.tensor(loss)):
                raise RuntimeError(f"Non-finite loss at step {trainer.global_step}: {loss}")
            losses.append(loss)
            if first_logged_loss is None:
                first_logged_loss = loss
            if trainer.global_step % 20 == 0 or trainer.global_step == 1:
                print(
                    f"  step={trainer.global_step:3d} "
                    f"epoch={epoch:2d} "
                    f"loss={loss:.4f} "
                    f"lr={metrics['lr']:.2e}"
                )
        if losses and losses[-1] < convergence_loss:
            break

    if not losses:
        raise RuntimeError("Smoke training executed zero steps")
    final_loss = losses[-1]
    assert first_logged_loss is not None
    print(f"Final: step={trainer.global_step} first_loss={first_logged_loss:.4f} final_loss={final_loss:.4f}")

    # Convergence + monotonic guards. These are what turn this script from a
    # smoke wire test into a real end-to-end pipeline check.
    if final_loss >= convergence_loss:
        raise RuntimeError(
            f"Smoke training did not converge: final_loss={final_loss:.4f} "
            f"(threshold={convergence_loss:.4f})"
        )
    if first_logged_loss * monotonic_ratio <= final_loss:
        raise RuntimeError(
            f"Loss did not decrease enough: first={first_logged_loss:.4f} "
            f"final={final_loss:.4f} (expected final < {first_logged_loss * monotonic_ratio:.4f})"
        )

    # Persist the checkpoint + tokenizer for smoke_infer to consume.
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (out_dir / "model.pt").resolve()
    tokenizer_path = out_dir / "tokenizer.json"
    # ``save_checkpoint`` joins relative paths onto ``save_dir``; pass the
    # absolute path to avoid the ``save_dir/save_dir/`` nesting that produces.
    trainer.save_checkpoint(str(checkpoint_path))
    tokenizer_path.write_text(json.dumps(tokenizer.to_json(), indent=2), encoding="utf-8")
    # Also persist a small run manifest so smoke_infer can know what shape of
    # model to reconstruct.
    manifest = {
        "vocab_size": vocab_size,
        "model_size": "tiny",
        "d_model": 64,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 128,
        "max_seq_len": 32,
        "pad_token_id": PAD_ID,
        "first_loss": first_logged_loss,
        "final_loss": final_loss,
        "steps": trainer.global_step,
        "data_path": str(data_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"SMOKE TRAIN PASSED: checkpoint={checkpoint_path}")
    return SmokeTrainResult(
        trainer=trainer,
        tokenizer=tokenizer,
        losses=losses,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end CPU smoke training round")
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp_manual/smoke_train"))
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--convergence-loss", type=float, default=0.15)
    parser.add_argument("--monotonic-ratio", type=float, default=0.15)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_smoke_train(
            data_path=args.data,
            out_dir=args.out_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            seed=args.seed,
            convergence_loss=args.convergence_loss,
            monotonic_ratio=args.monotonic_ratio,
        )
    except Exception as exc:  # pragma: no cover - exercised via CI failure signal
        print(f"SMOKE TRAIN FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
