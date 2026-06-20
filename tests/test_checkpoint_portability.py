"""Regression tests for portable, self-describing model checkpoints."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from lingolite.mobile_translation_model import (
    create_model,
    extract_model_state_dict,
    load_model_from_checkpoint,
)
from lingolite.training import TranslationTrainer, build_arg_parser
from lingolite.quantization_utils import (
    apply_dynamic_quantization,
    make_quantized_checkpoint,
)
from lingolite.utils import atomic_torch_save
from scripts.benchmark import _bench_decode
from scripts.export_mobile import LogitsOnlyWrapper


def _custom_model() -> torch.nn.Module:
    return create_model(
        vocab_size=48,
        model_size="tiny",
        d_model=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        pad_token_id=0,
    )


def _loader() -> DataLoader:
    batch = {
        "src_input_ids": torch.tensor([[4, 5, 2]], dtype=torch.long),
        "tgt_input_ids": torch.tensor([[1, 6, 2]], dtype=torch.long),
        "src_attention_mask": torch.ones(1, 3),
        "tgt_attention_mask": torch.ones(1, 3),
    }
    return DataLoader([batch], batch_size=None)


def test_trainer_checkpoint_rebuilds_custom_architecture(tmp_path: Path) -> None:
    model = _custom_model()
    trainer = TranslationTrainer(
        model=model,
        train_loader=_loader(),
        max_steps=2,
        warmup_steps=0,
        device="cpu",
        save_dir=str(tmp_path),
    )
    trainer.save_checkpoint("model.pt")

    checkpoint = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["config"] == model.get_config()

    restored = load_model_from_checkpoint(checkpoint)
    assert restored.get_config() == model.get_config()
    for key, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)


def test_extract_model_state_dict_normalizes_compiled_prefix() -> None:
    state = _custom_model().state_dict()
    compiled_style = {f"_orig_mod.{key}": value for key, value in state.items()}

    extracted = extract_model_state_dict({"model_state_dict": compiled_style})

    assert extracted.keys() == state.keys()


def test_checkpoint_loader_rejects_non_tensor_payload() -> None:
    with pytest.raises(ValueError, match="string keys to tensors"):
        extract_model_state_dict({"state_dict": {"weight": "not-a-tensor"}})


def test_mobile_export_wrapper_returns_logits_only() -> None:
    model = _custom_model().eval()
    wrapper = LogitsOnlyWrapper(model).eval()
    src = torch.tensor([[4, 5, 2]], dtype=torch.long)
    tgt = torch.tensor([[1, 6]], dtype=torch.long)
    src_mask = torch.ones_like(src)
    tgt_mask = torch.ones_like(tgt)

    with torch.inference_mode():
        expected, _, _ = model(src, tgt, src_mask, tgt_mask)
        actual = wrapper(src, tgt, src_mask, tgt_mask)

    torch.testing.assert_close(actual, expected)


def test_training_cli_exposes_memory_controls() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--train-data",
            "train.json",
            "--tokenizer-path",
            "tokenizer",
            "--gradient-accumulation-steps",
            "4",
            "--gradient-checkpointing",
        ]
    )

    assert args.gradient_accumulation_steps == 4
    assert args.gradient_checkpointing is True


def test_training_cli_exposes_resume_checkpoint() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--train-data",
            "train.json",
            "--tokenizer-path",
            "tokenizer",
            "--save-dir",
            "checkpoints",
            "--resume-from",
            "checkpoint_step_160000.pt",
        ]
    )

    assert args.resume_from == "checkpoint_step_160000.pt"


def test_decode_benchmark_reports_throughput() -> None:
    result = _bench_decode(
        name="greedy",
        model=_custom_model(),
        device=torch.device("cpu"),
        batch_size=1,
        src_len=4,
        max_length=4,
        vocab_size=48,
        sos_id=1,
        eos_id=2,
        pad_id=0,
        warmup=0,
        iters=1,
        seed=0,
        num_beams=2,
    )

    assert result.tokens_per_sec is not None
    assert result.tokens_per_sec > 0


def test_quantized_checkpoint_round_trip() -> None:
    model = _custom_model().eval()
    quantized = apply_dynamic_quantization(model, dtype=torch.qint8)
    checkpoint = make_quantized_checkpoint(quantized)

    restored = load_model_from_checkpoint(checkpoint)
    src = torch.tensor([[4, 5, 2]], dtype=torch.long)
    tgt = torch.tensor([[1, 6]], dtype=torch.long)

    with torch.inference_mode():
        expected, _, _ = quantized(src, tgt)
        actual, _, _ = restored(src, tgt)

    torch.testing.assert_close(actual, expected)


def test_atomic_torch_save_preserves_existing_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "checkpoint.pt"
    destination.write_bytes(b"previous-good-checkpoint")

    def fail_save(obj: object, path: Path) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(OSError, match="simulated write failure"):
        atomic_torch_save({"value": torch.tensor(1)}, destination)

    assert destination.read_bytes() == b"previous-good-checkpoint"
    assert list(tmp_path.glob("*.tmp")) == []
