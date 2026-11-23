"""YOLO object detection wrapper (Ultralytics, COCO-pretrained).

Only COCO classes relevant to road users are kept: person, bicycle, car, motorcycle, bus, truck.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COCO_ROAD = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE = {2, 5, 7}
TWO_WHEEL = {1, 3}


@dataclass
class Detection:
    frame_index: int
    cls: int
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]   # pixel coords in the input frame
    track_id: int = -1


class CheckpointError(FileNotFoundError):
    pass


def load_yolo(weights: str):
    from ultralytics import YOLO
    try:
        return YOLO(weights)
    except FileNotFoundError as e:
        raise CheckpointError(f"YOLO weights not found / not downloadable: {weights}") from e


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.5,
                 classes: list[int] | None = None, device: str = "cpu", imgsz: int = 640):
        self.model = load_yolo(weights)
        self.conf, self.iou, self.device, self.imgsz = conf, iou, device, imgsz
        self.classes = classes or list(COCO_ROAD)

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        res = self.model.predict(frame, conf=self.conf, iou=self.iou, classes=self.classes, device=self.device,
                                 imgsz=self.imgsz, verbose=False)[0]
        return parse_boxes(res, frame_index)


def parse_boxes(res, frame_index: int) -> list[Detection]:
    out: list[Detection] = []
    b = res.boxes
    if b is None or len(b) == 0:
        return out
    xyxy = b.xyxy.cpu().numpy()
    cls = b.cls.cpu().numpy().astype(int)
    conf = b.conf.cpu().numpy()
    ids = b.id.cpu().numpy().astype(int) if b.id is not None else -np.ones(len(cls), int)
    for i in range(len(cls)):
        out.append(Detection(frame_index, int(cls[i]), COCO_ROAD.get(int(cls[i]), str(cls[i])), float(conf[i]),
                             tuple(float(v) for v in xyxy[i]), int(ids[i])))
    return out
