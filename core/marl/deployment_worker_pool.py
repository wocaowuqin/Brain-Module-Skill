"""Bounded worker pool for causal, stateless deployment inference."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Any, Callable, Generic, Sequence, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class WorkerPoolStats:
    submitted: int
    completed: int
    rejected_backpressure: int
    failed: int
    in_flight: int


class StatelessHRLWorkerPool(Generic[T, R]):
    """Run pure batch inference on independent workers with backpressure.

    ``infer_batch`` must be stateless with respect to requests and resource
    accounting.  The pool never performs a ledger mutation; the caller must
    submit returned plans to the central exact transaction.  A bounded queue is
    intentional: rejecting at admission is preferable to unbounded latency.
    """

    def __init__(
        self,
        infer_batch: Callable[[Sequence[T]], R],
        *,
        worker_count: int = 3,
        max_in_flight: int = 6,
        thread_name_prefix: str = "hrl-worker",
    ) -> None:
        if worker_count <= 0 or max_in_flight <= 0:
            raise ValueError("worker_count and max_in_flight must be positive")
        self.infer_batch = infer_batch
        self.worker_count = int(worker_count)
        self.max_in_flight = int(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix=thread_name_prefix,
        )
        self._slots = threading.BoundedSemaphore(self.max_in_flight)
        self._lock = threading.Lock()
        self._submitted = self._completed = self._rejected = self._failed = 0
        self._closed = False

    def submit(self, batch: Sequence[T]) -> Future[R] | None:
        """Submit an arrived batch, or return ``None`` on backpressure."""

        if not batch:
            raise ValueError("batch must not be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("worker pool is closed")
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._rejected += 1
            return None
        with self._lock:
            self._submitted += 1
        try:
            future = self._executor.submit(self._run, tuple(batch))
        except BaseException:
            self._slots.release()
            with self._lock:
                self._submitted -= 1
                self._failed += 1
            raise
        future.add_done_callback(self._done)
        return future

    def _run(self, batch: tuple[T, ...]) -> R:
        try:
            return self.infer_batch(batch)
        except BaseException:
            with self._lock:
                self._failed += 1
            raise

    def _done(self, future: Future[R]) -> None:
        self._slots.release()
        with self._lock:
            self._completed += int(not future.cancelled() and future.exception() is None)

    def stats(self) -> WorkerPoolStats:
        with self._lock:
            return WorkerPoolStats(
                submitted=self._submitted,
                completed=self._completed,
                rejected_backpressure=self._rejected,
                failed=self._failed,
                in_flight=self._submitted - self._completed - self._failed,
            )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)


__all__ = ["StatelessHRLWorkerPool", "WorkerPoolStats"]

