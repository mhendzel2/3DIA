"""Background job execution utilities with progress and cancellation."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from pymaris.backends.registry import BackendRegistry
from pymaris.logging import get_logger
from pymaris.workflow import WorkflowCancelledError, WorkflowResult, WorkflowStep

LOGGER = get_logger(__name__)

ProgressCallback = Callable[[int, str], None]


class JobCancelledError(RuntimeError):
    """Raised when a background job is cancelled."""


TaskCallable = Callable[[ProgressCallback, threading.Event], Any]


@dataclass
class JobHandle:
    """Represents a submitted background job."""

    future: Future[Any]
    cancel_event: threading.Event

    def cancel(self) -> bool:
        self.cancel_event.set()
        return self.future.cancel()

    def done(self) -> bool:
        return self.future.done()

    def cancelled(self) -> bool:
        return self.future.cancelled() or self.cancel_event.is_set()

    def result(self, timeout: float | None = None) -> Any:
        return self.future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self.future.exception(timeout=timeout)


class JobRunner:
    """Run workflow steps in background threads with cooperative cancellation."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pymaris-job")

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def submit(
        self,
        step: WorkflowStep,
        context: Mapping[str, Any],
        *,
        registry: BackendRegistry | None = None,
        on_progress: ProgressCallback | None = None,
        resource_limits: Mapping[str, Any] | None = None,
    ) -> JobHandle:
        cancel_event = threading.Event()
        callback = on_progress or _noop_progress

        def worker() -> WorkflowResult:
            try:
                return step.run(
                    context=context,
                    registry=registry,
                    on_progress=callback,
                    cancel_event=cancel_event,
                    resource_limits=resource_limits,
                )
            except WorkflowCancelledError as exc:
                LOGGER.info("Workflow step cancelled: %s", step.id)
                raise JobCancelledError(str(exc)) from exc
            except Exception:
                LOGGER.exception("Workflow step failed: %s", step.id)
                raise

        future = self._executor.submit(worker)
        return JobHandle(future=future, cancel_event=cancel_event)

    def submit_callable(
        self,
        task: TaskCallable,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> JobHandle:
        """Submit an arbitrary long-running task with cooperative cancellation."""
        cancel_event = threading.Event()
        callback = on_progress or _noop_progress

        def worker() -> Any:
            try:
                return task(callback, cancel_event)
            except WorkflowCancelledError as exc:
                raise JobCancelledError(str(exc)) from exc
            except Exception:
                LOGGER.exception("Background task failed")
                raise

        future = self._executor.submit(worker)
        return JobHandle(future=future, cancel_event=cancel_event)


def _noop_progress(_: int, __: str) -> None:
    return None
