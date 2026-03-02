"""
Fallback CPU inference engine using ONNX Runtime.
Used when RKNN is unavailable or for host-side testing.
"""
import cv2, time, logging
import numpy as np
from typing import List, Tuple
from inference.rknn_engine import Detection, letterbox, dfl, decode_boxes, multiclass_nms
from inference.rknn_engine import OBJ_THRESH, NMS_THRESH, STRIDES

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    logger.warning("onnxruntime not available -- fallback engine disabled")


class FallbackEngine:
    """ONNX Runtime CPU fallback for host development/CI."""

    def __init__(self, onnx_path: str, class_names: List[str],
                 obj_thresh: float = OBJ_THRESH, nms_thresh: float = NMS_THRESH):
        self.class_names = class_names
        self.obj_thresh  = obj_thresh
        self.nms_thresh  = nms_thresh
        self._session = None
        if _ORT_AVAILABLE:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                onnx_path, sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info(f"[FallbackEngine] loaded {onnx_path}")

    def infer(self, img_bgr: np.ndarray) -> Tuple[List[Detection], float]:
        if self._session is None:
            return [], 0.0
        t0 = time.perf_counter()
        lb, ratio, (dw, dh) = letterbox(img_bgr)
        inp = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)   # NCHW
        outputs = self._session.run(None, {self._input_name: inp})
        # YOLOv8 ONNX output: [1, 4+nc, 8400] concatenated
        pred = outputs[0][0].T   # [8400, 4+nc]
        boxes_xywh = pred[:, :4]
        cls_logits = pred[:, 4:]
        scores = cls_logits.max(axis=1)
        cls_ids = cls_logits.argmax(axis=1)
        mask = scores >= self.obj_thresh
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]; cls_ids = cls_ids[mask]
        # xywh -> xyxy
        x1 = boxes_xywh[:,0] - boxes_xywh[:,2]/2
        y1 = boxes_xywh[:,1] - boxes_xywh[:,3]/2
        x2 = boxes_xywh[:,0] + boxes_xywh[:,2]/2
        y2 = boxes_xywh[:,1] + boxes_xywh[:,3]/2
        boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
        keep = multiclass_nms(boxes_xyxy, cls_ids, scores, self.nms_thresh)
        oh, ow = img_bgr.shape[:2]
        result = []
        for idx in keep:
            bx1 = float(np.clip((x1[idx]-dw)/ratio, 0, ow))
            by1 = float(np.clip((y1[idx]-dh)/ratio, 0, oh))
            bx2 = float(np.clip((x2[idx]-dw)/ratio, 0, ow))
            by2 = float(np.clip((y2[idx]-dh)/ratio, 0, oh))
            result.append(Detection(
                class_id=int(cls_ids[idx]),
                class_name=self.class_names[int(cls_ids[idx])],
                confidence=float(scores[idx]),
                bbox=[bx1,by1,bx2,by2],
                bbox_norm=[bx1/ow,by1/oh,bx2/ow,by2/oh]
            ))
        return result, (time.perf_counter()-t0)*1000.0
