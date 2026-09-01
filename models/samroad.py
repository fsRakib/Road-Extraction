"""
SAM-Road - Segment Anything (ViT-B) adapted to road graph extraction.

Outputs a connected road GRAPH, not a mask, so it does not suffer the
broken-connectivity problem that skeletonizing a mask always has. That
matters for you: OSRM needs connected ways, not pretty pixels.

Zero-shot here means: pretrained on CityScale (US/Australian cities, SAM
ViT-B encoder), never touched with a single Bangladesh image. No training
happens in this file - only loading two checkpoints and running inference.

Paper: https://arxiv.org/abs/2403.16051
Repo:  https://github.com/htcr/sam_road

SETUP (one time) - already done for you if you're reading this after the
"try samroad zero-shot" run, but written out for the record:

    git clone https://github.com/htcr/sam_road.git ~/sam_road
    git clone https://github.com/htcr/segment-anything-road.git ~/sam_road/sam

    mkdir -p ~/sam_road/sam_ckpts
    curl -L -o ~/sam_road/sam_ckpts/sam_vit_b_01ec64.pth \\
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

    curl -L -o models/weights/cityscale_vitb_512_e10.ckpt \\
      https://huggingface.co/congrui/sam_road/resolve/main/cityscale_vitb_512_e10.ckpt

    uv pip install lightning pytorch_lightning wandb rtree imageio \\
                    opencv-python-headless torchmetrics addict matplotlib

    export SAM_ROAD_DIR=~/sam_road

PERFORMANCE ON CPU
-------------------
The paper's default samples a densely overlapping 16x16 grid of 512x512
patches per 2048x2048 image (256 ViT-B forward passes) - built for a GPU.
On this CPU-only laptop that is very slow, so the grid density defaults to
6x6 (36 patches) here. Set SAMROAD_PATCHES_PER_EDGE=16 for the paper's full
density if you have time to spare, or lower for a quicker look.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as cfg
from models._base import RoadModel

SAM_ROAD_DIR = Path(os.environ.get("SAM_ROAD_DIR", Path.home() / "sam_road"))
SAM_ROAD_CONFIG = SAM_ROAD_DIR / "config" / "toponet_vitb_512_cityscale.yaml"
SAM_VIT_CKPT = SAM_ROAD_DIR / "sam_ckpts" / "sam_vit_b_01ec64.pth"
ROAD_CKPT = cfg.WEIGHTS / "cityscale_vitb_512_e10.ckpt"
PATCHES_PER_EDGE = int(os.environ.get("SAMROAD_PATCHES_PER_EDGE", "6"))


class SAMRoad(RoadModel):
    name = "samroad"
    description = "SAM ViT-B road graph, zero-shot on CityScale weights"
    outputs = "graph"

    def load(self):
        for path, hint in [
            (SAM_ROAD_DIR, "git clone https://github.com/htcr/sam_road.git ~/sam_road"),
            (SAM_ROAD_DIR / "sam" / "segment_anything",
             "git clone https://github.com/htcr/segment-anything-road.git ~/sam_road/sam"),
            (SAM_VIT_CKPT, "download sam_vit_b_01ec64.pth into ~/sam_road/sam_ckpts/"),
            (ROAD_CKPT, "download cityscale_vitb_512_e10.ckpt into models/weights/"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}\n  {hint}")

        sys.path.insert(0, str(SAM_ROAD_DIR))
        import torch
        from utils import load_config
        from model import SAMRoad as _Net
        import graph_extraction

        self.torch = torch
        self.graph_extraction = graph_extraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print(f"[samroad] no GPU - running a {PATCHES_PER_EDGE}x{PATCHES_PER_EDGE} "
                  "patch grid on CPU, this will take a few minutes")

        gconfig = load_config(str(SAM_ROAD_CONFIG))
        gconfig.SAM_CKPT_PATH = str(SAM_VIT_CKPT)   # repo default is a relative path
        self.gconfig = gconfig

        net = _Net(gconfig)
        checkpoint = torch.load(str(ROAD_CKPT), map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        net.to(self.device).eval()
        self.net = net
        return self

    def predict_graph(self, image):
        """image: (H, W, 3) uint8 RGB, same array extract.py already read from the GeoTIFF."""
        torch = self.torch
        gconfig = self.gconfig
        device = self.device

        from dataset import get_patch_info_one_img

        h, w = image.shape[:2]
        patch = gconfig.PATCH_SIZE
        margin = gconfig.SAMPLE_MARGIN
        all_patch_info = get_patch_info_one_img(0, h, margin, patch, PATCHES_PER_EDGE)

        batch_size = gconfig.INFER_BATCH_SIZE
        keypoint_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
        road_mask = torch.zeros((h, w), dtype=torch.float32, device=device)
        counter = torch.zeros((h, w), dtype=torch.float32, device=device)
        img_features = []

        n_batches = (len(all_patch_info) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            info = all_patch_info[bi * batch_size:(bi + 1) * batch_size]
            patches = np.stack([image[y0:y1, x0:x1] for _, (x0, y0), (x1, y1) in info])
            batch = torch.tensor(patches, dtype=torch.float32, device=device)

            with torch.no_grad():
                mask_scores, feats = self.net.infer_masks_and_img_features(batch)
            img_features.append(feats)

            for i, (_, (x0, y0), (x1, y1)) in enumerate(info):
                keypoint_mask[y0:y1, x0:x1] += mask_scores[i, :, :, 0]
                road_mask[y0:y1, x0:x1] += mask_scores[i, :, :, 1]
                counter[y0:y1, x0:x1] += 1.0
            print(f"[samroad]   pass 1/2 (masks): patch batch {bi + 1}/{n_batches}")

        counter = torch.clamp(counter, min=1.0)
        keypoint_mask = (keypoint_mask / counter * 255).to(torch.uint8).cpu().numpy()
        road_mask = (road_mask / counter * 255).to(torch.uint8).cpu().numpy()

        points = self.graph_extraction.extract_graph_points(keypoint_mask, road_mask, gconfig)
        if points.shape[0] == 0:
            return []

        import rtree
        import scipy.spatial

        tree = rtree.index.Index()
        for i, (x, y) in enumerate(points):
            tree.insert(i, (x, y, x, y))

        edge_scores, edge_counts = defaultdict(float), defaultdict(float)
        for bi in range(n_batches):
            info = all_patch_info[bi * batch_size:(bi + 1) * batch_size]
            topo = {"points": [], "pairs": [], "valid": []}
            idx_maps = []

            for _, (x0, y0), (x1, y1) in info:
                idxs = list(tree.intersection((x0, y0, x1, y1)))
                idx_maps.append(dict(enumerate(idxs)))
                local = points[idxs] - np.array([[x0, y0]])
                n = len(idxs)
                if n == 0:
                    topo["points"].append(local)
                    topo["pairs"].append(np.zeros((0, gconfig.MAX_NEIGHBOR_QUERIES, 2), int))
                    topo["valid"].append(np.zeros((0, gconfig.MAX_NEIGHBOR_QUERIES), bool))
                    continue
                kdt = scipy.spatial.KDTree(local)
                _, knn = kdt.query(local, k=gconfig.MAX_NEIGHBOR_QUERIES + 1,
                                   distance_upper_bound=gconfig.NEIGHBOR_RADIUS)
                knn = np.atleast_2d(knn)[:, 1:]
                src = np.tile(np.arange(n)[:, None], (1, gconfig.MAX_NEIGHBOR_QUERIES))
                valid = knn < n
                tgt = np.where(valid, knn, src)
                topo["points"].append(local)
                topo["pairs"].append(np.stack([src, tgt], axis=-1))
                topo["valid"].append(valid)

            max_n = max(x.shape[0] for x in topo["points"])
            if max_n == 0:
                continue
            collated = {
                k: np.stack([np.pad(x, [(0, max_n - x.shape[0])] + [(0, 0)] * (x.ndim - 1))
                            for x in v], axis=0)
                for k, v in topo.items()
            }

            with torch.no_grad():
                scores = self.net.infer_toponet(
                    img_features[bi],
                    torch.tensor(collated["points"], device=device),
                    torch.tensor(collated["pairs"], device=device),
                    torch.tensor(collated["valid"], device=device),
                )
            scores = torch.where(torch.isnan(scores), torch.tensor(-100.0), scores)
            scores = scores.squeeze(-1).cpu().numpy()

            for si in range(scores.shape[0]):
                for pi_ in range(scores.shape[1]):
                    for pj in range(scores.shape[2]):
                        if not collated["valid"][si, pi_, pj]:
                            continue
                        a, b = collated["pairs"][si, pi_, pj]
                        a, b = idx_maps[si].get(int(a)), idx_maps[si].get(int(b))
                        if a is None or b is None:
                            continue
                        key = (a, b)
                        edge_scores[key] += float(scores[si, pi_, pj])
                        edge_counts[key] += 1.0
            print(f"[samroad]   pass 2/2 (topology): patch batch {bi + 1}/{n_batches}")

        paths = []
        for (a, b), total in edge_scores.items():
            if total / edge_counts[(a, b)] > gconfig.TOPO_THRESHOLD:
                xa, ya = points[a]
                xb, yb = points[b]
                paths.append([(int(ya), int(xa)), (int(yb), int(xb))])
        return paths
