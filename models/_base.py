"""
The one interface every model must follow.

To add a model, create models/<yourmodel>.py containing:

    from models._base import RoadModel

    class YourModel(RoadModel):
        name = "yourmodel"          # used in output filenames

        def load(self):
            ...                     # load weights once

        def predict(self, patch):
            ...                     # return float32 HxW, values 0..1

Nothing else in the project needs to change.
"""
import numpy as np


class RoadModel:
    name = "unnamed"
    description = ""

    def load(self):
        """Called once before the first predict(). Load weights here."""
        return self

    def predict(self, patch: np.ndarray) -> np.ndarray:
        """
        patch  : uint8 array, shape (H, W, 3), RGB
        returns: float32 array, shape (H, W), 0.0 = not road, 1.0 = road
        """
        raise NotImplementedError
