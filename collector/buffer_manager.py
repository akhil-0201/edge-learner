"""
buffer_manager.py - Thread-safe frame buffer for edge-learner.

Collects annotated frames that fall below confidence threshold
for later use in dataset building and fine-tuning.
"""
from __future__ import annotations

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    source: str
    detections: list  # raw detection dicts from the inference pipeline
    variant: str = "unknown"


class BufferManager:
    """
    Thread-safe ring-buffer that accumulates low-confidence frames.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames kept in memory before the oldest
        are evicted.
    flush_interval : float
        Seconds between automatic disk flushes (0 = manual only).
    save_dir : str | Path
        Directory where flushed frames are persisted as JPEG files.
    """

    def __init__(
        self,
        max_frames: int = 500,
        flush_interval: float = 60.0,
        save_dir: str | Path = "/var/lib/edge-learner/buffer",
    ) -> None:
        self._max_frames = max_frames
        self._flush_interval = flush_interval
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: Deque[Frame] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if flush_interval > 0:
            self._flush_thread = threading.Thread(
                target=self._auto_flush_loop,
                name="BufferManager-flush",
                daemon=True,
            )
            self._flush_thread.start()
        else:
            self._flush_thread = None

        logger.info(
            "BufferManager started (max_frames=%d, flush_interval=%.1fs, save_dir=%s)",
            max_frames, flush_interval, self._save_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, frame: Frame) -> None:
        """Add a frame to the buffer (thread-safe)."""
        with self._lock:
            self._buffer.append(frame)

    def drain(self) -> List[Frame]:
        """
        Remove and return all frames currently in the buffer.
        Safe to call from any thread.
        """
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames

    def peek(self) -> List[Frame]:
        """Return a snapshot of the buffer without clearing it."""
        with self._lock:
            return list(self._buffer)

    def size(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    def flush_to_disk(self) -> int:
        """
        Persist all buffered frames to *save_dir* and clear the buffer.

        Returns
        -------
        int
            Number of frames written.
        """
        frames = self.drain()
        if not frames:
            return 0

        written = 0
        for frm in frames:
            ts = int(frm.timestamp * 1000)
            filename = self._save_dir / f"{frm.variant}_{frm.source}_{ts}.jpg"
            try:
                cv2.imwrite(str(filename), frm.image)
                written += 1
            except Exception as exc:
                logger.warning("Failed to write frame %s: %s", filename, exc)

        logger.info("Flushed %d frames to %s", written, self._save_dir)
        return written

    def stop(self) -> None:
        """Stop the background flush thread and do a final flush."""
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5)
        self.flush_to_disk()
        logger.info("BufferManager stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_flush_loop(self) -> None:
        """Background thread: flush to disk every *flush_interval* seconds."""
        while not self._stop_event.wait(timeout=self._flush_interval):
            try:
                self.flush_to_disk()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Auto-flush error: %s", exc)
