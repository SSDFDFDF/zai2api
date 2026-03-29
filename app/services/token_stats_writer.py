"""Async batch writer for token usage statistics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.utils.logger import logger, log_exception


@dataclass(frozen=True)
class TokenStatsUpdate:
    token: str
    token_id: int
    successful_requests: int = 0
    failed_requests: int = 0


class AsyncTokenStatsWriter:
    """Queue-based batch writer for token stats."""

    def __init__(
        self,
        *,
        batch_size: int = 200,
        flush_interval: float = 1.0,
        max_queue_size: int = 20000,
    ) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: asyncio.Queue[TokenStatsUpdate] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._task: Optional[asyncio.Task] = None
        self._flush_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="token-stats-writer")
        logger.debug("AsyncTokenStatsWriter started")

    async def stop(self) -> None:
        if self._task is not None:
            await self._cancel_worker(restart=False)
        else:
            await self._drain_queue()
        logger.debug("AsyncTokenStatsWriter stopped")

    def enqueue(
        self,
        *,
        token: str,
        token_id: int,
        successful_requests: int = 0,
        failed_requests: int = 0,
    ) -> bool:
        if successful_requests <= 0 and failed_requests <= 0:
            return True

        try:
            self._queue.put_nowait(
                TokenStatsUpdate(
                    token=token,
                    token_id=token_id,
                    successful_requests=successful_requests,
                    failed_requests=failed_requests,
                )
            )
            return True
        except asyncio.QueueFull:
            logger.warning("Token stats queue full, skipping async enqueue")
            return False

    async def flush_now(self) -> None:
        if self._task is not None:
            await self._cancel_worker(restart=True)
            return
        await self._drain_queue()

    async def _drain_queue(self) -> None:
        async with self._flush_lock:
            while not self._queue.empty():
                batch = await self._collect_batch(wait_for_first=False)
                if not batch:
                    break
                await asyncio.shield(self._write_batch(batch))

    async def _cancel_worker(self, *, restart: bool) -> None:
        task = self._task
        if task is None:
            if restart:
                await self.start()
            return

        self._task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await self._drain_queue()
        if restart:
            await self.start()

    async def _run(self) -> None:
        while True:
            batch: List[TokenStatsUpdate] = []
            try:
                batch = await self._collect_batch(wait_for_first=True)
                if batch:
                    async with self._flush_lock:
                        await asyncio.shield(self._write_batch(batch))
            except asyncio.CancelledError:
                if batch:
                    async with self._flush_lock:
                        await asyncio.shield(self._write_batch(batch))
                raise
            except Exception as exc:
                log_exception(logger, "AsyncTokenStatsWriter flush error")
                await asyncio.sleep(1)

    async def _collect_batch(self, *, wait_for_first: bool) -> List[TokenStatsUpdate]:
        batch: List[TokenStatsUpdate] = []
        if wait_for_first:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval
                )
                batch.append(item)
            except asyncio.TimeoutError:
                return batch

        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    @staticmethod
    def _merge_batch(
        batch: List[TokenStatsUpdate],
    ) -> Dict[int, Dict[str, Any]]:
        merged: Dict[int, Dict[str, Any]] = {}
        for item in batch:
            entry = merged.setdefault(
                item.token_id,
                {
                    "token": item.token,
                    "successful_requests": 0,
                    "failed_requests": 0,
                },
            )
            entry["successful_requests"] += item.successful_requests
            entry["failed_requests"] += item.failed_requests
        return merged

    @staticmethod
    async def _write_batch(batch: List[TokenStatsUpdate]) -> None:
        from app.services.token_dao import get_token_dao
        from app.utils.token_pool import get_token_pool

        merged = AsyncTokenStatsWriter._merge_batch(batch)
        dao = get_token_dao()
        await dao.record_stats_batch(merged)

        pool = get_token_pool()
        if pool:
            for entry in merged.values():
                await pool.mark_stats_synced(
                    token=str(entry["token"]),
                    successful_requests=int(entry["successful_requests"]),
                    failed_requests=int(entry["failed_requests"]),
                )


_token_stats_writer: Optional[AsyncTokenStatsWriter] = None


def get_token_stats_writer() -> AsyncTokenStatsWriter:
    global _token_stats_writer
    if _token_stats_writer is None:
        _token_stats_writer = AsyncTokenStatsWriter()
    return _token_stats_writer
