from models.engine.model import Model
from models import DetectionModel
from models.detect import DetectionInfer


class AlgModel(Model):
    """Local object detection model."""

    def __init__(self, model="pretrained_11l_nocls.pt", task=None, verbose=False):
        """Initialize model"""
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "infer": DetectionInfer,
            }
        }
