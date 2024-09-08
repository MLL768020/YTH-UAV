# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer,DetectionTrainer1
from .val import DetectionValidator,DetectionValidator1

__all__ = 'DetectionPredictor', 'DetectionTrainer', 'DetectionValidator','DetectionTrainer1','DetectionValidator1'
