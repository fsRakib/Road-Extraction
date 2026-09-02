"""
Plain U-Net (Ronneberger et al.) - 4-level encoder/decoder, skip connections,
no pretrained backbone. Trained on the Massachusetts Roads Dataset.

Weights: models/weights/unet_road_massachusetts.pth
Source:  https://huggingface.co/teohyc/Satellite-Road-Segmentation-UNet
         (checkpoint verified tensor-by-tensor against this exact class - 136
         tensors, all keys and shapes match)

IMPORTANT - resolution mismatch, read before trusting the results
-------------------------------------------------------------------
The training pipeline resizes each full 1500x1500 px tile (1 m/px Massachusetts
aerial imagery) down to 256x256 before feeding the network - i.e. every image
the network has ever seen is effectively ~5.9 m/px, no matter how sharp the
source photo was. That is far coarser than our 0.55 m/px Bangladesh tiles.

To match that scale as closely as this pipeline's patch-based design allows,
each 512x512 patch (0.55 m/px, ~280 m across) is resized down to 256x256
before inference - about 1.1 m/px, still roughly 5x sharper than what the
network was trained on. In practice this means it may see our roads as
unusually thin and under-detect them; unlike dlinknet, this is not a strong
model to rely on for Bangladesh imagery. See README.md.
"""
import numpy as np
import torch
import torch.nn as nn

import config
from models._base import RoadModel

WEIGHTS = config.WEIGHTS / "unet_road_massachusetts.pth"
NET_SIZE = 256   # fixed by how the checkpoint was trained


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return torch.sigmoid(self.conv_final(d1))


class UNetRoad(RoadModel):
    name = "unet"
    description = "Plain U-Net, Massachusetts Roads Dataset (scale mismatch - see file docstring)"

    def load(self):
        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Missing weights: {WEIGHTS}\n"
                "Download best_road_seg_unet.pth from "
                "https://huggingface.co/teohyc/Satellite-Road-Segmentation-UNet"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)

        self.net = UNet(in_channels=3, out_channels=1)
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()
        return self

    def predict(self, patch):
        ph, pw = patch.shape[:2]
        x = torch.from_numpy(patch.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)                          # 1x3xHxW
        x = torch.nn.functional.interpolate(x, size=(NET_SIZE, NET_SIZE),
                                            mode="bilinear", align_corners=False)
        x = x.to(self.device)

        with torch.no_grad():
            out = self.net(x)
            out = torch.nn.functional.interpolate(out, size=(ph, pw),
                                                  mode="bilinear", align_corners=False)
        return out[0, 0].cpu().numpy().astype(np.float32)
