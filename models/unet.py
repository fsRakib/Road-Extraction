"""
U-Net template (segmentation_models_pytorch).

Setup:
    pip install torch segmentation-models-pytorch
    put your trained weights at  models/weights/unet_road.pth

Copy this file to add D-LinkNet, SAMRoad, etc. - only WEIGHTS,
name, and the network line usually change.
"""
import numpy as np

import config
from models._base import RoadModel

WEIGHTS = config.WEIGHTS / "unet_road.pth"


class UNetRoad(RoadModel):
    name = "unet"
    description = "U-Net, resnet34 encoder"

    def load(self):
        import torch
        import segmentation_models_pytorch as smp

        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Missing weights: {WEIGHTS}\n"
                "Download a road-segmentation checkpoint and save it there, "
                "or run with --model baseline for now."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = smp.Unet("resnet34", encoder_weights=None,
                            in_channels=3, classes=1)
        self.net.load_state_dict(torch.load(WEIGHTS, map_location=self.device))
        self.net.to(self.device).eval()
        self.torch = torch
        return self

    def predict(self, patch):
        torch = self.torch
        # ImageNet normalisation - the encoder was trained with it
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        x = (patch.astype(np.float32) / 255.0 - mean) / std
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)

        with torch.no_grad():
            logits = self.net(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return prob.astype(np.float32)
