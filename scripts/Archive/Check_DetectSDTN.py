import os
import sys
import torch

# 1) ensure project root is on sys.path
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

# 2) import your modules
from ultralytics.nn.modules.block import SpatialTransformer
from ultralytics.nn.modules.head  import DetectSDTN

if __name__ == "__main__":
    # Instantiate STN with c1=3 channels
    stn = SpatialTransformer(3)    # ← correct constructor usage :contentReference[oaicite:1]{index=1}

    # build a tiny backbone
    backbone = torch.nn.Sequential(
        stn,
        torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
        torch.nn.ReLU(),
    )

    # DetectSDTN expects ch=(32,32,32) if you have three P-levels of 32 channels each
    head = DetectSDTN(nc=1, ch=(32, 32, 32))

    imgs = torch.randn(2, 3, 640, 640)
    feat = backbone(imgs)                # → (2,32,160,160)
    feats = [feat, feat, feat]           # P3, P4, P5 all the same for smoke-test

    # forward through head alone
    out = head(feats)


    if isinstance(out, list):
        print("OK – returned list, first entry shape:", out[0].shape)
    else:
        print("OK – returned tensor shape:", out.shape)
