"""Bounded async microbatch scheduling, independent of the HTTP transport."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Deque, List, Optional, Sequence

from lingolite.inference import GenerationOptions, TranslationEngine, TranslationInput, TranslationResult


class ServiceBusyError(RuntimeError):
    """The replica has reached its outstanding-request limit."""


class ServiceUnavailableError(RuntimeError):
    """The replica has not started or is shutting down."""


@dataclass
class _Pending:
    item: TranslationInput
    options: GenerationOptions
    future: asyncio.Future


class TranslationService:
    """One model worker per replica, with bounded admission and compatible batches.

    Limits include active work. Client cancellation/timeout never frees an active
    slot until its compute actually finishes. No GPU tensor is allocated on admission.
    """

    def __init__(self, engine: TranslationEngine, *, max_batch_size: int = 8,
                 max_pending: int = 128, max_batch_tokens: int = 8192,
                 batch_wait_ms: float = 5.0, request_timeout: float = 30.0) -> None:
        for name, value in (("max_batch_size", max_batch_size), ("max_pending", max_pending),
                            ("max_batch_tokens", max_batch_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be finite and non-negative")
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_pending = max_pending
        self.max_batch_tokens = max_batch_tokens
        self.batch_wait_ms = batch_wait_ms
        self.request_timeout = request_timeout
        self._pending: Deque[_Pending] = deque()
        self._outstanding = 0
        self._event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._closed = False
        self.completed_batches = 0

    @property
    def ready(self) -> bool:
        return not self._closed and self._task is not None and not self._task.done()

    @property
    def outstanding(self) -> int:
        return self._outstanding

    async def start(self) -> None:
        if self._closed:
            raise ServiceUnavailableError("A closed translation service cannot restart")
        if self._task is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingolite")
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        self._closed = True
        while self._pending:
            job = self._pending.popleft()
            self._outstanding -= 1
            if not job.future.done():
                job.future.set_exception(ServiceUnavailableError("Translation service is shutting down"))
        self._event.set()
        if self._task is not None:
            await self._task
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    async def translate(self, item: TranslationInput, options: GenerationOptions) -> TranslationResult:
        return (await self.translate_many([item], options))[0]

    async def translate_many(self, inputs: Sequence[TranslationInput],
                             options: GenerationOptions) -> List[TranslationResult]:
        if not self.ready:
            raise ServiceUnavailableError("Translation service is not ready")
        if not inputs:
            raise ValueError("At least one input is required")
        if self.engine.token_cost(options) > self.max_batch_tokens:
            raise ValueError("Generation exceeds the replica's batch token budget")
        # Atomic admission on the event loop, including requests already executing.
        if self._outstanding + len(inputs) > self.max_pending:
            raise ServiceBusyError("Translation queue is full; retry later")
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in inputs]
        self._pending.extend(_Pending(item, options, future) for item, future in zip(inputs, futures))
        self._outstanding += len(inputs)
        self._event.set()
        results = await asyncio.wait_for(
            asyncio.gather(*futures, return_exceptions=True), timeout=self.request_timeout,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    def _take_batch(self) -> List[_Pending]:
        # Discard cancelled queued work; active slots are released by _run only.
        live = deque()
        for job in self._pending:
            if job.future.done():
                self._outstanding -= 1
            else:
                live.append(job)
        self._pending = live
        if not live:
            return []
        first = live.popleft()
        batch = [first]
        limit = min(self.max_batch_size, self.max_batch_tokens // self.engine.token_cost(first.options))
        remaining = deque()
        for job in live:
            if job.options == first.options and len(batch) < limit:
                batch.append(job)
            else:
                remaining.append(job)
        self._pending = remaining
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            await self._event.wait()
            self._event.clear()
            if self.batch_wait_ms:
                await asyncio.sleep(self.batch_wait_ms / 1000)
            while self._pending and not self._closed:
                batch = self._take_batch()
                if not batch:
                    break
                try:
                    results = await loop.run_in_executor(
                        self._executor, self.engine.translate_batch,
                        [job.item for job in batch], batch[0].options,
                    )
                    if len(results) != len(batch):
                        raise RuntimeError("Translation engine returned an unexpected batch size")
                    self.completed_batches += 1
                    for job, result in zip(batch, results):
                        if not job.future.done():
                            if isinstance(result, Exception):
                                job.future.set_exception(result)
                            else:
                                job.future.set_result(result)
                except Exception as exc:
                    for job in batch:
                        if not job.future.done():
                            job.future.set_exception(exc)
                finally:
                    self._outstanding -= len(batch)
