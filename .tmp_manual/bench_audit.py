"""Audit benchmark: train-step, greedy-KV-cache, beam-search latency + memory + a simple
attention-mask sync profiling.

Run on CPU (no CUDA in this environment).
"""
from __future__ import annotations

import time
import gc
import tracemalloc
from statistics import median

import torch

from lingolite.mobile_translation_model import create_model


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_train_step(model, vocab_size: int, batch: int, src_len: int, tgt_len: int, iters: int = 25) -> dict:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    src = torch.randint(1, vocab_size, (batch, src_len))
    tgt = torch.randint(1, vocab_size, (batch, tgt_len))
    src_mask = torch.ones(batch, src_len)
    tgt_mask = torch.ones(batch, tgt_len)
    # Warmup
    for _ in range(3):
        opt.zero_grad()
        loss = model.compute_loss(src, tgt, src_mask, tgt_mask, label_smoothing=0.1)
        loss.backward()
        opt.step()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = model.compute_loss(src, tgt, src_mask, tgt_mask, label_smoothing=0.1)
        loss.backward()
        opt.step()
        sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {"median_ms": median(times) * 1000.0, "mean_ms": sum(times) * 1000.0 / len(times)}


def bench_greedy(model, vocab_size: int, batch: int, src_len: int, max_length: int, iters: int = 10) -> dict:
    model.eval()
    src = torch.randint(5, vocab_size, (batch, src_len))
    src_mask = torch.ones(batch, src_len)
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate(src, src_mask, max_length=max_length, sos_token_id=1, eos_token_id=2, num_beams=1)
    sync()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model.generate(src, src_mask, max_length=max_length, sos_token_id=1, eos_token_id=2, num_beams=1)
            sync()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return {"median_ms": median(times) * 1000.0, "mean_ms": sum(times) * 1000.0 / len(times)}


def bench_beam(model, vocab_size: int, batch: int, src_len: int, max_length: int, beams: int, iters: int = 5) -> dict:
    model.eval()
    src = torch.randint(5, vocab_size, (batch, src_len))
    src_mask = torch.ones(batch, src_len)
    with torch.no_grad():
        for _ in range(2):
            _ = model.generate_beam(src, src_mask, max_length=max_length, num_beams=beams, sos_token_id=1, eos_token_id=2)
    sync()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model.generate_beam(src, src_mask, max_length=max_length, num_beams=beams, sos_token_id=1, eos_token_id=2)
            sync()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return {"median_ms": median(times) * 1000.0, "mean_ms": sum(times) * 1000.0 / len(times)}


def peak_mem_train_step(model, vocab_size: int, batch: int, src_len: int, tgt_len: int) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    src = torch.randint(1, vocab_size, (batch, src_len))
    tgt = torch.randint(1, vocab_size, (batch, tgt_len))
    src_mask = torch.ones(batch, src_len)
    tgt_mask = torch.ones(batch, tgt_len)
    gc.collect()
    tracemalloc.start()
    opt.zero_grad()
    loss = model.compute_loss(src, tgt, src_mask, tgt_mask, label_smoothing=0.1)
    loss.backward()
    opt.step()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024.0 * 1024.0)


def main() -> None:
    torch.manual_seed(0)
    vocab_size = 8000
    model = create_model(vocab_size=vocab_size, model_size="tiny")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M (tiny)")
    print()
    # Train step
    r1 = bench_train_step(model, vocab_size, batch=4, src_len=32, tgt_len=32, iters=10)
    print(f"train_step  B=4 src=32 tgt=32 : median={r1['median_ms']:.1f} ms")
    peak = peak_mem_train_step(model, vocab_size, batch=4, src_len=32, tgt_len=32)
    print(f"  peak python-allocated mem: {peak:.1f} MiB (tracemalloc, python side)")
    print()
    # Greedy
    r2 = bench_greedy(model, vocab_size, batch=1, src_len=16, max_length=32)
    print(f"greedy KV   B=1 src=16 gen=32 : median={r2['median_ms']:.1f} ms")
    r3 = bench_greedy(model, vocab_size, batch=4, src_len=32, max_length=48)
    print(f"greedy KV   B=4 src=32 gen=48 : median={r3['median_ms']:.1f} ms")
    print()
    # Beam
    r4 = bench_beam(model, vocab_size, batch=1, src_len=16, max_length=24, beams=4)
    print(f"beam search B=1 src=16 gen=24 beams=4 : median={r4['median_ms']:.1f} ms")
    r5 = bench_beam(model, vocab_size, batch=2, src_len=20, max_length=24, beams=4)
    print(f"beam search B=2 src=20 gen=24 beams=4 : median={r5['median_ms']:.1f} ms")


if __name__ == "__main__":
    main()
