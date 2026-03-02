import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
  # Model
  model_path: str = os.getenv("MODEL_PATH", "/opt/edge-learner/models/current.rknn")
  model_version: str = os.getenv("MODEL_VERSION", "v1.0.0")
  variant_id: str = os.getenv("VARIANT_ID", "default")
  num_cores: int = int(os.getenv("NPU_CORE_COUNT", "3"))

  # Inference
  confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
  low_conf_threshold: float = float(os.getenv("LOW_CONF_THRESHOLD", "0.3"))
  iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
  input_size: int = int(os.getenv("INPUT_SIZE", "640"))
  class_names_path: str = os.getenv("CLASS_NAMES_PATH", "/opt/edge-learner/models/classes.txt")

  # Camera
  camera_source: str = os.getenv("CAMERA_SOURCE", "0")
  camera_width: int = int(os.getenv("CAMERA_WIDTH", "1280"))
  camera_height: int = int(os.getenv("CAMERA_HEIGHT", "720"))
  camera_fps: int = int(os.getenv("CAMERA_FPS", "30"))

  # Buffer
  buffer_dir: str = os.getenv("BUFFER_DIR", "/opt/edge-learner/buffer")
  buffer_max_size: int = int(os.getenv("BUFFER_MAX_SIZE", "500"))
  jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "85"))

  # Trainer
  training_trigger_count: int = int(os.getenv("TRAINING_TRIGGER_COUNT", "100"))
  training_epochs: int = int(os.getenv("TRAINING_EPOCHS", "50"))
  training_batch_size: int = int(os.getenv("TRAINING_BATCH_SIZE", "16"))
  base_model_path: str = os.getenv("BASE_MODEL_PATH", "/opt/edge-learner/models/base.pt")
  training_data_dir: str = os.getenv("TRAINING_DATA_DIR", "/opt/edge-learner/training_data")
  export_dir: str = os.getenv("EXPORT_DIR", "/opt/edge-learner/exports")
  rknn_toolkit_path: str = os.getenv("RKNN_TOOLKIT_PATH", "/opt/rknn-toolkit2")

  # Uploader
  upload_endpoint: str = os.getenv("UPLOAD_ENDPOINT", "")
  upload_api_key: str = os.getenv("UPLOAD_API_KEY", "")
  upload_timeout: int = int(os.getenv("UPLOAD_TIMEOUT", "120"))

  # Logging
  log_level: str = os.getenv("LOG_LEVEL", "INFO")
  log_dir: str = os.getenv("LOG_DIR", "/opt/edge-learner/logs")

  def __post_init__(self):
    for d in [self.buffer_dir, self.training_data_dir, self.export_dir, self.log_dir]:
      Path(d).mkdir(parents=True, exist_ok=True)

  @property
  def class_names(self) -> list:
    p = Path(self.class_names_path)
    if p.exists():
      return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return []


_settings: Optional[Settings] = None


def get_settings() -> Settings:
  global _settings
  if _settings is None:
    _settings = Settings()
  return _settings
