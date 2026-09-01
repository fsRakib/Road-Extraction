"""
D-LinkNet34 - winner of the DeepGlobe 2018 Road Extraction Challenge.

ResNet34 encoder + a dilated centre block + LinkNet decoder. The dilated
block is the whole idea: it widens the receptive field without losing
resolution, so the network can follow a road that disappears under trees
and reappears further along.

Paper:   https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf
Weights: models/weights/dlinknet34_deepglobe.th  (trained on DeepGlobe, 0.5 m/px)

Trained on rural/suburban roads in Thailand, India and Indonesia, so it
transfers to Bangladesh better than any city-trained model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

import config
from models._base import RoadModel

WEIGHTS = config.WEIGHTS / "dlinknet34_deepglobe.th"


class Dblock(nn.Module):
    """Dilated centre block: four 3x3 convs with dilation 1, 2, 4, 8, summed."""

    def __init__(self, ch):
        super().__init__()
        self.dilate1 = nn.Conv2d(ch, ch, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(ch, ch, 3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(ch, ch, 3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(ch, ch, 3, dilation=8, padding=8)

    def forward(self, x):
        d1 = F.relu(self.dilate1(x))
        d2 = F.relu(self.dilate2(d1))
        d3 = F.relu(self.dilate3(d2))
        d4 = F.relu(self.dilate4(d3))
        return x + d1 + d2 + d3 + d4


class DecoderBlock(nn.Module):
    """1x1 squeeze -> transposed conv (x2) -> 1x1 expand."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = in_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid, 1)
        self.norm1 = nn.BatchNorm2d(mid)
        self.deconv2 = nn.ConvTranspose2d(mid, mid, 3, stride=2,
                                          padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1)
        self.norm3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.deconv2(x)))
        return F.relu(self.norm3(self.conv3(x)))


class DinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        res = resnet34(weights=None)
        self.firstconv = res.conv1
        self.firstbn = res.bn1
        self.firstrelu = res.relu
        self.firstmaxpool = res.maxpool
        self.encoder1, self.encoder2 = res.layer1, res.layer2
        self.encoder3, self.encoder4 = res.layer3, res.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.firstmaxpool(self.firstrelu(self.firstbn(self.firstconv(x))))
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.dblock(self.encoder4(e3))

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = F.relu(self.finaldeconv1(d1))
        out = F.relu(self.finalconv2(out))
        return torch.sigmoid(self.finalconv3(out))


class DLinkNet34(RoadModel):
    name = "dlinknet"
    description = "D-LinkNet34, DeepGlobe winner (rural roads)"

    tta = True    # 4-way flip averaging: meaningfully better, ~4x slower - fine on CPU

    def load(self):
        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"Missing weights: {WEIGHTS}\n"
                "Download log01_dink34.th from the D-LinkNet repo and save it there."
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        # the checkpoint was saved from nn.DataParallel, so keys start with "module."
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        self.net = DinkNet34()
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()
        torch.set_num_threads(max(1, torch.get_num_threads()))
        return self

    def _prep(self, patch):
        # the original repo read images with cv2 (BGR) and scaled to [-1.6, 1.6]
        bgr = patch[:, :, ::-1].astype(np.float32)
        return bgr / 255.0 * 3.2 - 1.6

    def predict(self, patch):
        x = self._prep(patch)
        batch = [x]
        if self.tta:
            batch = [x, x[::-1], x[:, ::-1], x[::-1, ::-1]]

        arr = np.ascontiguousarray(np.stack(batch).transpose(0, 3, 1, 2))
        with torch.no_grad():
            out = self.net(torch.from_numpy(arr).to(self.device))
        out = out[:, 0].cpu().numpy()

        if self.tta:
            out = np.stack([out[0], out[1][::-1], out[2][:, ::-1],
                            out[3][::-1, ::-1]]).mean(0)
        else:
            out = out[0]
        return out.astype(np.float32)
