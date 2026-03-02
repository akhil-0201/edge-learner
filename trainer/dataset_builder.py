import logging
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class DatasetBuilder:
  """
  Builds a YOLO-format dataset from low-confidence buffered frames.
  Supports pseudo-labelling via a reference/teacher model pass.
  """

  YAML_TEMPLATE = """
path: {dataset_path}
train: images/train
val: images/val
nc: {nc}
names: {names}
"""

  def __init__(
    self,
    buffer_dir: str,
    output_dir: str,
    class_names: List[str],
    val_split: float = 0.15,
  ):
    self.buffer_dir = Path(buffer_dir)
    self.output_dir = Path(output_dir)
    self.class_names = class_names
    self.val_split = val_split

  def build(self, metas: List[Dict]) -> Optional[Path]:
    """
    Build dataset from buffer metadata entries.
    Returns path to dataset yaml or None on failure.
    """
    if not metas:
      logger.warning("No metadata entries to build dataset from")
      return None

    dataset_path = self.output_dir / "dataset"
    for split in ["train", "val"]:
      (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
      (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(len(metas) * self.val_split))
    val_metas = metas[:n_val]
    train_metas = metas[n_val:]

    written = 0
    for split, split_metas in [("val", val_metas), ("train", train_metas)]:
      for meta in split_metas:
        ok = self._write_sample(meta, dataset_path, split)
        if ok:
          written += 1

    if written == 0:
      logger.error("No samples written to dataset")
      return None

    yaml_path = dataset_path / "data.yaml"
    yaml_path.write_text(
      self.YAML_TEMPLATE.format(
        dataset_path=str(dataset_path),
        nc=len(self.class_names),
        names=self.class_names,
      )
    )

    logger.info(f"Dataset built: {written} samples at {dataset_path}")
    return yaml_path

  def _write_sample(
    self,
    meta: Dict,
    dataset_path: Path,
    split: str,
  ) -> bool:
    try:
      src_img = Path(meta["image_path"])
      if not src_img.exists():
        return False

      frame_id = meta["frame_id"]
      dst_img = dataset_path / "images" / split / f"{frame_id}.jpg"
      shutil.copy2(src_img, dst_img)

      labels = meta.get("inference_labels", [])
      label_lines = []
      for det in labels:
        cls_name = det.get("class", "")
        if cls_name not in self.class_names:
          continue
        cls_idx = self.class_names.index(cls_name)
        bbox = det.get("bbox", {})
        cx = bbox.get("cx", 0.5)
        cy = bbox.get("cy", 0.5)
        w = bbox.get("w", 0.1)
        h = bbox.get("h", 0.1)
        label_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

      dst_label = dataset_path / "labels" / split / f"{frame_id}.txt"
      dst_label.write_text("\n".join(label_lines))
      return True
    except Exception as e:
      logger.warning(f"Failed to write sample {meta.get('frame_id')}: {e}")
      return False

  def cleanup(self):
    if self.output_dir.exists():
      shutil.rmtree(self.output_dir / "dataset", ignore_errors=True)
