"""
Real-time frame pipeline: Camera -> preprocess -> infer -> collect.
"""
import cv2, time, threading, logging, queue
from typing import List
from .rknn_engine import RKNNEngine, Detection

logger = logging.getLogger(__name__)


class FramePipeline:
    def __init__(self, engine: RKNNEngine, monitor,
                 source=0, target_fps: float = 30.0):
        self.engine     = engine
        self.monitor    = monitor
        self.source     = source
        self.target_fps = target_fps
        self._frame_q   = queue.Queue(maxsize=4)
        self._stop      = threading.Event()
        self._stats     = {"frames": 0, "latency_ms": [], "fps": 0.0}

    def start(self):
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._infer_loop,   daemon=True).start()
        logger.info("[FramePipeline] started")

    def stop(self):
        self._stop.set()

    def _capture_loop(self):
        cap    = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        period = 1.0 / self.target_fps
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret: continue
            if not self._frame_q.full():
                self._frame_q.put_nowait(frame)
            time.sleep(max(0, period - (time.perf_counter()-t0)))
        cap.release()

    def _infer_loop(self):
        fps_t0, fps_n = time.time(), 0
        while not self._stop.is_set():
            try:
                frame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
            dets, lat = self.engine.infer(frame)
            self._stats["latency_ms"].append(lat)
            if len(self._stats["latency_ms"]) > 300:
                self._stats["latency_ms"].pop(0)
            fps_n += 1
            if time.time() - fps_t0 >= 1.0:
                self._stats["fps"] = fps_n / (time.time()-fps_t0)
                fps_n = 0; fps_t0 = time.time()
            self.monitor.on_frame(frame, dets)

    @property
    def stats(self):
        import numpy as np
        lats = self._stats["latency_ms"]
        if lats:
            self._stats["p50_ms"] = float(np.percentile(lats, 50))
            self._stats["p95_ms"] = float(np.percentile(lats, 95))
            self._stats["p99_ms"] = float(np.percentile(lats, 99))
        return self._stats
