"""
Stage 1: Low-confidence frame detection and buffering.
"""
import time, threading, logging
from typing import List

logger = logging.getLogger(__name__)


class ConfidenceMonitor:
    def __init__(self, cfg: dict, buffer, tagger):
        self.conf_low   = cfg["conf_low_threshold"]
        self.conf_high  = cfg["conf_high_threshold"]
        self.rate_limit = cfg["collection_rate_limit_sec"]
        self.buffer     = buffer
        self.tagger     = tagger
        self._last      = 0.0
        self._lock      = threading.Lock()
        self._stats     = {"collected": 0, "skipped_rate": 0, "skipped_conf": 0}

    def on_frame(self, frame, detections: List):
        """Called from inference loop -- non-blocking."""
        if not detections: return
        uncertain = [d for d in detections
                     if self.conf_low <= d.confidence <= self.conf_high]
        if not uncertain:
            self._stats["skipped_conf"] += 1; return
        now = time.time()
        with self._lock:
            if now - self._last < self.rate_limit:
                self._stats["skipped_rate"] += 1; return
            self._last = now
        threading.Thread(
            target=self.buffer.write,
            args=(frame, uncertain, self.tagger.current_variant),
            daemon=True
        ).start()
        self._stats["collected"] += 1

    @property
    def stats(self): return dict(self._stats)
