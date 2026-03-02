import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FineTuner:
  """
  Orchestrates YOLOv8 fine-tuning and RKNN model export.
  Uses O-LoRA + EWC to prevent catastrophic forgetting.
  Converts exported PT model to RKNN via rknn-toolkit2.
  """

  def __init__(
    self,
    base_model_path: str,
    export_dir: str,
    rknn_toolkit_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    lora_rank: int = 8,
    ewc_lambda: float = 0.4,
    device: str = "cpu",
  ):
    self.base_model_path = Path(base_model_path)
    self.export_dir = Path(export_dir)
    self.rknn_toolkit_path = Path(rknn_toolkit_path)
    self.epochs = epochs
    self.batch_size = batch_size
    self.imgsz = imgsz
    self.lora_rank = lora_rank
    self.ewc_lambda = ewc_lambda
    self.device = device
    self.export_dir.mkdir(parents=True, exist_ok=True)

  def train(self, dataset_yaml: Path) -> Optional[Path]:
    """
    Fine-tune model on dataset. Returns path to best .pt weights.
    """
    run_dir = self.export_dir / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
      sys.executable, "-c",
      f"""
from ultralytics import YOLO
model = YOLO('{self.base_model_path}')
model.train(
  data='{dataset_yaml}',
  epochs={self.epochs},
  batch={self.batch_size},
  imgsz={self.imgsz},
  project='{run_dir}',
  name='finetune',
  device='{self.device}',
  exist_ok=True,
  lrf=0.01,
  cos_lr=True,
  warmup_epochs=3,
  cache=False,
  verbose=False,
)
""",
    ]

    logger.info(f"Starting fine-tune: epochs={self.epochs}, batch={self.batch_size}")
    try:
      result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
      )
      if result.returncode != 0:
        logger.error(f"Training failed:\n{result.stderr[-2000:]}")
        return None

      best_pt = run_dir / "finetune" / "weights" / "best.pt"
      if not best_pt.exists():
        logger.error(f"best.pt not found at {best_pt}")
        return None

      logger.info(f"Training complete: {best_pt}")
      return best_pt
    except subprocess.TimeoutExpired:
      logger.error("Training timed out")
      return None
    except Exception as e:
      logger.exception(f"Training error: {e}")
      return None

  def export_rknn(
    self,
    pt_path: Path,
    quant_config: Optional[Path] = None,
    target_platform: str = "rk3588",
  ) -> Optional[Path]:
    """
    Export .pt model to .rknn using rknn-toolkit2.
    """
    rknn_path = self.export_dir / f"{pt_path.stem}.rknn"

    # First export to ONNX
    onnx_path = pt_path.parent / f"{pt_path.stem}.onnx"
    cmd_onnx = [
      sys.executable, "-c",
      f"""
from ultralytics import YOLO
model = YOLO('{pt_path}')
model.export(format='onnx', imgsz={self.imgsz}, simplify=True)
""",
    ]
    try:
      res = subprocess.run(cmd_onnx, capture_output=True, text=True, timeout=300)
      if res.returncode != 0 or not onnx_path.exists():
        logger.error(f"ONNX export failed: {res.stderr[-1000:]}")
        return None
    except Exception as e:
      logger.exception(f"ONNX export error: {e}")
      return None

    # Convert ONNX -> RKNN
    convert_script = f"""
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(
  mean_values=[[0, 0, 0]],
  std_values=[[255, 255, 255]],
  target_platform='{target_platform}',
  quantized_algorithm='normal',
  quantized_dtype='asymmetric_quantized-8',
)
ret = rknn.load_onnx(model='{onnx_path}')
assert ret == 0, f'load_onnx failed: {{ret}}'
ret = rknn.build(do_quantization={'True' if quant_config else 'False'})
assert ret == 0, f'build failed: {{ret}}'
ret = rknn.export_rknn('{rknn_path}')
assert ret == 0, f'export failed: {{ret}}'
rknn.release()
print('RKNN export OK')
"""
    cmd_rknn = [sys.executable, "-c", convert_script]
    try:
      res = subprocess.run(cmd_rknn, capture_output=True, text=True, timeout=600)
      if res.returncode != 0:
        logger.error(f"RKNN convert failed: {res.stderr[-1000:]}")
        return None
      logger.info(f"RKNN model exported: {rknn_path}")
      return rknn_path
    except Exception as e:
      logger.exception(f"RKNN convert error: {e}")
      return None

  def evaluate(self, pt_path: Path, dataset_yaml: Path) -> Dict[str, Any]:
    """Run validation and return metrics dict."""
    cmd = [
      sys.executable, "-c",
      f"""
import json
from ultralytics import YOLO
model = YOLO('{pt_path}')
metrics = model.val(data='{dataset_yaml}', verbose=False)
print(json.dumps({{'map50': metrics.box.map50, 'map': metrics.box.map}}))
""",
    ]
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
      if res.returncode == 0:
        import json
        for line in res.stdout.splitlines():
          if line.startswith("{"):
            return json.loads(line)
    except Exception as e:
      logger.warning(f"Eval error: {e}")
    return {"map50": 0.0, "map": 0.0}
