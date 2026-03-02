"""
RKNN Inference Engine -- YOLOv8 on RK3588 NPU
Ref: https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py
    https://github.com/airockchip/rknn_model_zoo/blob/main/py_utils/rknn_executor.py
"""
import cv2, time, logging, threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

IMG_SIZE   = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
STRIDES    = [8, 16, 32]
REG_MAX    = 16

try:
    from rknnlite.api import RKNNLite
    _RKNN_AVAILABLE = True
except ImportError:
    _RKNN_AVAILABLE = False
    logger.warning("rknnlite not installed -- engine in stub mode")


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       List[float]   # [x1, y1, x2, y2] original coords
    bbox_norm:  List[float]   # [0,1] normalised


def letterbox(img, size=IMG_SIZE, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(size/h, size/w)
    nw, nh = int(round(w*r)), int(round(h*r))
    dw, dh = (size-nw)/2, (size-nh)/2
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    t, b = int(round(dh-0.1)), int(round(dh+0.1))
    l, r_ = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, t, b, l, r_, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def dfl(pos):
    n, c, h, w = pos.shape
    p = 4; mc = c // p
    y = pos.reshape(n, p, mc, h, w)
    e = np.exp(y - y.max(axis=2, keepdims=True))
    y = e / e.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1,1,mc,1,1)
    return (y * acc).sum(axis=2)


def decode_boxes(reg, stride):
    _, _, gh, gw = reg.shape
    gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing='ij')
    xy = reg[:,:2] + np.stack([gx,gy])[None]
    xy = xy * stride
    wh = np.exp(reg[:,2:]) * stride
    return np.concatenate([xy - wh/2, xy + wh/2], axis=1)


def multiclass_nms(boxes, classes, scores, iou_thresh=NMS_THRESH):
    keep_all = []
    for c in np.unique(classes):
        mask = classes == c
        b, s = boxes[mask], scores[mask]
        x1,y1,x2,y2 = b[:,0],b[:,1],b[:,2],b[:,3]
        areas = (x2-x1)*(y2-y1)
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
            iou = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order = order[np.where(iou<=iou_thresh)[0]+1]
        keep_all.extend(np.where(mask)[0][keep].tolist())
    return keep_all


class RKNNEngine:
    """
    Thread-safe RKNN inference engine wrapping rknnlite.api.RKNNLite.
    Ref: https://github.com/airockchip/rknn-toolkit2/blob/master/rknn-toolkit-lite2/README.md
    """
    def __init__(self, model_path, class_names, obj_thresh=OBJ_THRESH, nms_thresh=NMS_THRESH):
        self.model_path  = model_path
        self.class_names = class_names
        self.obj_thresh  = obj_thresh
        self.nms_thresh  = nms_thresh
        self._lock = threading.Lock()
        self._rknn = None
        self._load()

    def _load(self):
        if not _RKNN_AVAILABLE: return
        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(self.model_path)
        if ret != 0: raise RuntimeError(f"load_rknn failed (ret={ret})")
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0: raise RuntimeError(f"init_runtime failed (ret={ret})")
        logger.info(f"[RKNNEngine] loaded {self.model_path}")

    def release(self):
        if self._rknn: self._rknn.release(); self._rknn = None

    def reload(self, new_path):
        new_rknn = RKNNLite(verbose=False)
        ret = new_rknn.load_rknn(new_path)
        if ret != 0: raise RuntimeError(f"reload failed: {new_path}")
        ret = new_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0: raise RuntimeError("init_runtime failed on reload")
        with self._lock:
            old = self._rknn; self._rknn = new_rknn; self.model_path = new_path
            if old: old.release()
        logger.info(f"[RKNNEngine] reloaded -> {new_path}")

    def infer(self, img_bgr):
        t0 = time.perf_counter()
        lb, ratio, (dw, dh) = letterbox(img_bgr)
        inp = np.expand_dims(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB), 0)
        with self._lock:
            if self._rknn is None: return [], 0.0
            outputs = self._rknn.inference(inputs=[inp], data_format="nhwc")
        dets = self._decode(outputs, ratio, dw, dh, img_bgr.shape[:2])
        return dets, (time.perf_counter()-t0)*1000.0

    def _decode(self, outputs, ratio, dw, dh, orig_shape):
        nc = len(self.class_names)
        all_boxes, all_cls, all_scores = [], [], []
        for i, stride in enumerate(STRIDES):
            box_out = dfl(outputs[i*2])
            box_out = decode_boxes(box_out, stride)
            cls_out = 1/(1+np.exp(-outputs[i*2+1]))
            gh, gw = box_out.shape[2:]
            boxes   = box_out[0].transpose(1,2,0).reshape(-1,4)
            clsmap  = cls_out[0].transpose(1,2,0).reshape(-1,nc)
            cls_ids = np.argmax(clsmap, axis=1)
            scores  = clsmap[np.arange(len(cls_ids)), cls_ids]
            mask    = scores >= self.obj_thresh
            all_boxes.append(boxes[mask]); all_cls.append(cls_ids[mask]); all_scores.append(scores[mask])
        if not any(len(b) for b in all_boxes): return []
        boxes  = np.concatenate(all_boxes)
        classes= np.concatenate(all_cls)
        scores = np.concatenate(all_scores)
        keep   = multiclass_nms(boxes, classes, scores, self.nms_thresh)
        oh, ow = orig_shape
        result = []
        for idx in keep:
            x1 = float(np.clip((boxes[idx,0]-dw)/ratio, 0, ow))
            y1 = float(np.clip((boxes[idx,1]-dh)/ratio, 0, oh))
            x2 = float(np.clip((boxes[idx,2]-dw)/ratio, 0, ow))
            y2 = float(np.clip((boxes[idx,3]-dh)/ratio, 0, oh))
            result.append(Detection(
                class_id=int(classes[idx]), class_name=self.class_names[int(classes[idx])],
                confidence=float(scores[idx]), bbox=[x1,y1,x2,y2],
                bbox_norm=[x1/ow,y1/oh,x2/ow,y2/oh]))
        return result
