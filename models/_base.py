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

    # "mask"  -> implement predict();       the pipeline skeletonizes for you
    # "graph" -> implement predict_graph(); the model gives lines directly
    outputs = "mask"

    def load(self):
        """Called once before the first predict(). Load weights here."""
        return self

    def predict(self, patch: np.ndarray) -> np.ndarray:
        """
        Mask models implement this. Called once per 512x512 patch.

        patch  : uint8 array, shape (H, W, 3), RGB
        returns: float32 array, shape (H, W), 0.0 = not road, 1.0 = road
        """
        raise NotImplementedError

    def predict_graph(self, image: np.ndarray):
        """
        Graph models implement this instead. Called once on the whole image.

        image  : uint8 array, shape (H, W, 3), RGB
        returns: list of paths, each a list of (row, col) pixel pairs
        """
        raise NotImplementedError
