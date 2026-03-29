"""Async batch log writer.

Accepts log entries via an asyncio.Queue and flushes them to the database
in batches (every ``flush_interval`` seconds or when ``batch_size`` entries
accumulate), reducing per-request SQLite write-lock contention.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from app.models.db_models import RequestLog
from app.utils.logger import logger, log_exception


class AsyncLogWriter:
    """Queue-based batch writer for request logs."""

    def __init__(
        self,
        *,
        batch_size: int = 50,
        flush_interval: float = 2.0,
        max_queue_size: int = 10000,
    ) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: asyncio.Queue[RequestLog] = asyncio.Queue(maxsize=max_queue_size)
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="log-writer")
        logger.debug("AsyncLogWriter started")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        # Drain remaining entries
        await self._flush_all()
        logger.debug("AsyncLogWriter stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, log_entry: RequestLog) -> None:
        """Non-blocking enqueue; drops entry on full queue."""
        try:
            self._queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            logger.warning("Log queue full, dropping entry")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._write_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_exception(logger, "AsyncLogWriter flush error")
                await asyncio.sleep(1)

    async def _collect_batch(self) -> List[RequestLog]:
        """Wait for items up to flush_interval or batch_size."""
        batch: List[RequestLog] = []
        try:
            # Block until at least one item arrives
            item = await asyncio.wait_for(
                self._queue.get(), timeout=self._flush_interval
            )
            batch.append(item)
        except asyncio.TimeoutError:
            return batch

        # Drain up to batch_size without waiting
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _flush_all(self) -> None:
        """Drain every remaining item in the queue."""
        batch: List[RequestLog] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._write_batch(batch)

    @staticmethod
    async def _write_batch(batch: List[RequestLog]) -> None:
        from app.database import get_db_session

        try:
            async with get_db_session() as session:
                session.add_all(batch)
                await session.commit()
        except Exception as exc:
            log_exception(logger, "Batch log write failed (%d entries)", len(batch))


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_log_writer: Optional[AsyncLogWriter] = None


def get_log_writer() -> AsyncLogWriter:
    global _log_writer
    if _log_writer is None:
        _log_writer = AsyncLogWriter()
    return _log_writer
