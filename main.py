#!/usr/bin/env python3
"""
edge-learner: Production-grade edge AI self-learning pipeline.
Entry point for the continuous learning daemon.
"""
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from collector.buffer_manager import BufferManager
from collector.confidence_monitor import ConfidenceMonitor
from inference.rknn_engine import RKNNEngine
from inference.frame_pipeline import FramePipeline
from trainer.dataset_builder import DatasetBuilder
from trainer.fine_tuner import FineTuner
from uploader.model_uploader import ModelUploader


def setup_logging(level: str, log_dir: str):
  log_path = Path(log_dir) / "edge-learner.log"
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
      logging.StreamHandler(sys.stdout),
      logging.FileHandler(str(log_path)),
    ],
  )


class EdgeLearner:
  """
  Main orchestrator. Runs two threads:
  1. Inference loop: captures frames, runs detection, collects low-confidence samples
  2. Training loop: periodically fine-tunes on collected data and hot-swaps model
  """

  def __init__(self):
    self.cfg = get_settings()
    self.logger = logging.getLogger(self.__class__.__name__)
    self._stop_event = threading.Event()
    self._model_lock = threading.Lock()

    self.engine = RKNNEngine(
      model_path=self.cfg.model_path,
      num_cores=self.cfg.num_cores,
    )
    self.buffer = BufferManager(
      buffer_dir=self.cfg.buffer_dir,
      max_size=self.cfg.buffer_max_size,
      jpeg_quality=self.cfg.jpeg_quality,
    )
    self.monitor = ConfidenceMonitor(
      buffer_manager=self.buffer,
      low_conf_threshold=self.cfg.low_conf_threshold,
      training_trigger_count=self.cfg.training_trigger_count,
    )
    self.pipeline = FramePipeline(
      engine=self.engine,
      confidence_threshold=self.cfg.confidence_threshold,
      input_size=self.cfg.input_size,
      class_names=self.cfg.class_names,
    )
    self.fine_tuner = FineTuner(
      base_model_path=self.cfg.base_model_path,
      export_dir=self.cfg.export_dir,
      rknn_toolkit_path=self.cfg.rknn_toolkit_path,
      epochs=self.cfg.training_epochs,
      batch_size=self.cfg.training_batch_size,
      imgsz=self.cfg.input_size,
    )
    self.uploader = ModelUploader(
      endpoint=self.cfg.upload_endpoint,
      api_key=self.cfg.upload_api_key,
      timeout=self.cfg.upload_timeout,
    )

  def start(self):
    self.logger.info("Starting edge-learner pipeline")
    if not self.engine.load():
      self.logger.error("Failed to load RKNN model. Aborting.")
      sys.exit(1)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, self._handle_signal)
    signal.signal(signal.SIGINT, self._handle_signal)

    inference_thread = threading.Thread(
      target=self._inference_loop,
      name="inference",
      daemon=True,
    )
    training_thread = threading.Thread(
      target=self._training_loop,
      name="training",
      daemon=True,
    )

    inference_thread.start()
    training_thread.start()

    self.logger.info("Edge-learner running. Press Ctrl+C to stop.")
    try:
      while not self._stop_event.is_set():
        time.sleep(1)
    finally:
      self.stop()

  def stop(self):
    self.logger.info("Shutting down edge-learner")
    self._stop_event.set()
    self.engine.release()

  def _handle_signal(self, signum, frame):
    self.logger.info(f"Received signal {signum}. Stopping...")
    self._stop_event.set()

  def _inference_loop(self):
    """Capture frames, run inference, feed monitor."""
    import cv2
    cam_src = self.cfg.camera_source
    try:
      src = int(cam_src)
    except ValueError:
      src = cam_src

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
    cap.set(cv2.CAP_PROP_FPS, self.cfg.camera_fps)

    self.logger.info(f"Inference loop started on camera: {src}")

    while not self._stop_event.is_set():
      ret, frame = cap.read()
      if not ret:
        self.logger.warning("Camera read failed, retrying...")
        time.sleep(0.1)
        continue

      with self._model_lock:
        detections = self.pipeline.process(frame)

      self.monitor.observe(
        frame=frame,
        detections=detections,
        variant_id=self.cfg.variant_id,
        model_version=self.cfg.model_version,
      )

    cap.release()
    self.logger.info("Inference loop stopped")

  def _training_loop(self):
    """Wait for training trigger, fine-tune, and hot-swap model."""
    self.logger.info("Training loop started")

    while not self._stop_event.is_set():
      if not self.monitor.should_train():
        time.sleep(10)
        continue

      self.logger.info("Training triggered")
      metas = self.buffer.read_all()
      if len(metas) < 10:
        self.logger.warning(f"Only {len(metas)} samples — waiting for more")
        time.sleep(30)
        continue

      # Build dataset
      builder = DatasetBuilder(
        buffer_dir=self.cfg.buffer_dir,
        output_dir=self.cfg.export_dir + "/tmp_dataset",
        class_names=self.cfg.class_names,
      )
      yaml_path = builder.build(metas)
      if yaml_path is None:
        self.logger.error("Dataset build failed")
        continue

      # Train
      pt_path = self.fine_tuner.train(yaml_path)
      if pt_path is None:
        self.logger.error("Training failed")
        builder.cleanup()
        continue

      # Evaluate
      metrics = self.fine_tuner.evaluate(pt_path, yaml_path)
      self.logger.info(f"Eval metrics: {metrics}")

      # Export RKNN
      rknn_path = self.fine_tuner.export_rknn(pt_path)
      if rknn_path is None:
        self.logger.warning("RKNN export failed — keeping current model")
        builder.cleanup()
        continue

      # Hot-swap model
      self._swap_model(rknn_path)

      # Upload
      model_id = self.uploader.upload(
        rknn_path,
        metadata={
          "variant_id": self.cfg.variant_id,
          "model_version": self.cfg.model_version,
          "metrics": metrics,
        },
      )
      if model_id:
        self.uploader.notify_deployment(
          model_id=model_id,
          variant_id=self.cfg.variant_id,
          metrics=metrics,
        )

      # Clear buffer after successful training
      self.monitor.reset()
      builder.cleanup()
      self.logger.info("Training cycle complete")

  def _swap_model(self, new_rknn_path: Path):
    """Atomically swap the active RKNN model."""
    self.logger.info(f"Swapping model to {new_rknn_path}")
    with self._model_lock:
      self.engine.release()
      self.engine.model_path = new_rknn_path
      ok = self.engine.load()
      if not ok:
        self.logger.error("New model failed to load — attempting reload of original")
        self.engine.model_path = Path(self.cfg.model_path)
        self.engine.load()


def main():
  cfg = get_settings()
  setup_logging(cfg.log_level, cfg.log_dir)
  learner = EdgeLearner()
  learner.start()


if __name__ == "__main__":
  main()
