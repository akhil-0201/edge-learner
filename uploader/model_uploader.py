import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Dict
import requests

logger = logging.getLogger(__name__)


class ModelUploader:
  """
  Uploads a trained RKNN model to a remote model registry.
  Supports retry, checksum validation, and metadata tagging.
  """

  def __init__(
    self,
    endpoint: str,
    api_key: str,
    timeout: int = 120,
    max_retries: int = 3,
    retry_delay: float = 5.0,
  ):
    self.endpoint = endpoint.rstrip("/")
    self.api_key = api_key
    self.timeout = timeout
    self.max_retries = max_retries
    self.retry_delay = retry_delay

  def upload(
    self,
    model_path: Path,
    metadata: Optional[Dict] = None,
  ) -> Optional[str]:
    """
    Upload model file to registry.
    Returns the remote model URL/ID on success, None on failure.
    """
    if not model_path.exists():
      logger.error(f"Model file not found: {model_path}")
      return None

    if not self.endpoint:
      logger.warning("No upload endpoint configured — skipping upload")
      return None

    checksum = self._sha256(model_path)
    meta = {
      "filename": model_path.name,
      "checksum_sha256": checksum,
      "size_bytes": model_path.stat().st_size,
      **(metadata or {}),
    }

    for attempt in range(1, self.max_retries + 1):
      try:
        logger.info(f"Uploading {model_path.name} (attempt {attempt}/{self.max_retries})")
        with open(model_path, "rb") as f:
          resp = requests.post(
            f"{self.endpoint}/models/upload",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"model": (model_path.name, f, "application/octet-stream")},
            data={"metadata": str(meta)},
            timeout=self.timeout,
          )

        if resp.status_code == 200:
          result = resp.json()
          model_id = result.get("model_id") or result.get("id")
          logger.info(f"Upload successful: {model_id}")
          return model_id
        else:
          logger.warning(f"Upload attempt {attempt} failed: HTTP {resp.status_code} — {resp.text[:200]}")

      except requests.exceptions.Timeout:
        logger.warning(f"Upload attempt {attempt} timed out")
      except requests.exceptions.ConnectionError as e:
        logger.warning(f"Upload attempt {attempt} connection error: {e}")
      except Exception as e:
        logger.exception(f"Upload attempt {attempt} unexpected error: {e}")

      if attempt < self.max_retries:
        time.sleep(self.retry_delay)

    logger.error(f"All {self.max_retries} upload attempts failed for {model_path.name}")
    return None

  def notify_deployment(
    self,
    model_id: str,
    variant_id: str,
    metrics: Optional[Dict] = None,
  ) -> bool:
    """Notify server that model has been deployed on this device."""
    if not self.endpoint or not model_id:
      return False
    try:
      resp = requests.post(
        f"{self.endpoint}/models/{model_id}/deployed",
        headers={"Authorization": f"Bearer {self.api_key}"},
        json={"variant_id": variant_id, "metrics": metrics or {}},
        timeout=30,
      )
      return resp.status_code == 200
    except Exception as e:
      logger.warning(f"Deployment notification failed: {e}")
      return False

  @staticmethod
  def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
      for chunk in iter(lambda: f.read(65536), b""):
        h.update(chunk)
    return h.hexdigest()
