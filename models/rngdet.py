"""
RNGDet++ - transformer that traces road graphs step by step.

Instead of segmenting pixels, it starts at a road and repeatedly asks
"where does this road go next?", like a car driving the network. The output
is already a routable vector graph, which is the closest thing here to what
OSRM actually wants.

Paper: https://arxiv.org/abs/2209.10150
Repo:  https://github.com/TonyXuQAQ/RNGDetPlusPlus

SETUP (one time)
----------------
    git clone https://github.com/TonyXuQAQ/RNGDetPlusPlus.git ~/RNGDetPlusPlus
    # download their pretrained checkpoint into models/weights/
    export RNGDET_DIR=~/RNGDetPlusPlus

WARNING - read before trying
----------------------------
RNGDet++ is AUTOREGRESSIVE: it runs one transformer forward pass per step
along every road, thousands of steps per image. On a GPU that is minutes.
On this CPU-only laptop it is realistically hours per image.

Do not start here. Use `dlinknet` for day-to-day work and come back to this
once you have a GPU machine.
"""
import os
import sys
from pathlib import Path

import numpy as np

import config
from models._base import RoadModel

RNGDET_DIR = Path(os.environ.get("RNGDET_DIR", Path.home() / "RNGDetPlusPlus"))
CHECKPOINT = config.WEIGHTS / "rngdetpp_best.pt"


class RNGDetPlusPlus(RoadModel):
    name = "rngdet"
    description = "RNGDet++ graph tracing (GPU only - hours on CPU)"
    outputs = "graph"

    def load(self):
        if not RNGDET_DIR.exists():
            raise FileNotFoundError(
                f"RNGDet++ repo not found at {RNGDET_DIR}\n"
                "  git clone https://github.com/TonyXuQAQ/RNGDetPlusPlus.git\n"
                "  export RNGDET_DIR=~/RNGDetPlusPlus\n"
                "See the setup notes at the top of models/rngdet.py."
            )
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Missing {CHECKPOINT} - download the checkpoint from the repo."
            )

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RNGDet++ needs a GPU. On CPU one image takes hours.\n"
                "Use 'dlinknet' instead, or run this on a GPU machine."
            )

        sys.path.insert(0, str(RNGDET_DIR))
        from models.detr import build_model  # from the cloned repo

        self.torch = torch
        self.device = "cuda"
        self.net = build_model()
        self.net.load_state_dict(torch.load(CHECKPOINT, map_location=self.device))
        self.net.to(self.device).eval()
        return self

    def predict_graph(self, image):
        from inference import trace_graph  # from the cloned repo

        nodes, edges = trace_graph(self.net, image, self.device)
        return [[tuple(nodes[a]), tuple(nodes[b])] for a, b in edges]
