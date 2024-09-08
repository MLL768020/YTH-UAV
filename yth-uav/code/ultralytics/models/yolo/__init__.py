# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment

from .model import YOLO,YOLO1

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO','YOLO1'
