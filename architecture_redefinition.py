#############################################
#  🔧 ARCHITECTURE DEFINITION FOR TESTING  #
#############################################

import torch
import torch.nn as nn
from transformers import RobertaModel, SwinModel

# Same as in training
NUM_CLASSES = {
    "t1":  2,
    "t2":  3,
    "t3t": 2,
    "t3s": 3,
    "t4":  3
}

class HEAD(nn.Module):
    def __init__(self, d, o):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, o)
        )
    def forward(self, x):
        return self.m(x)

class FUSE(nn.Module):
    def __init__(self, d=512, layers=2, heads=8):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            batch_first=True,
            dim_feedforward=d * 4,
            dropout=0.1,
            activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d))

    def forward(self, t, v):
        B = t.size(0)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, t[:, None], v[:, None]], dim=1)  # [B, 3, d]
        out = self.enc(seq)
        return out[:, 0]  # CLS token

class MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        # Text encoder
        self.txt = RobertaModel.from_pretrained("roberta-base")
        self.txt.pooler = None  # disable pooler

        # Vision encoder
        self.vis = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        # Projections to common 512-dim space
        self.tp = nn.Linear(self.txt.config.hidden_size, 512)
        self.vp = nn.Linear(self.vis.config.hidden_size, 512)

        # Fusion and shared trunk
        self.fuse = FUSE(512, layers=2, heads=8)
        self.shared_norm = nn.LayerNorm(512)

        # Extra capacity for t3 (damage sub-tasks)
        self.t3_gate = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512)
        )

        # Task-specific heads
        self.h1  = HEAD(512, 2)  # t1
        self.h2  = HEAD(512, 3)  # t2
        self.h3t = HEAD(512, 2)  # t3t
        self.h3s = HEAD(512, 3)  # t3s
        self.h4  = HEAD(512, 3)  # t4

    def forward(self, B):
        # Text CLS
        txt_out = self.txt(B["input_ids"], B["attention_mask"])
        t = txt_out.last_hidden_state[:, 0]  # [B, H_txt]

        # Visual pooled
        vis_out = self.vis(pixel_values=B["pixel_values"])
        v = vis_out.last_hidden_state.mean(1)  # [B, H_vis]

        # Project
        t_proj = self.tp(t)  # [B, 512]
        v_proj = self.vp(v)  # [B, 512]

        # Fuse
        z = self.fuse(t_proj, v_proj)  # [B, 512]
        z = self.shared_norm(z)

        # Special branch for t3 tasks
        z_t3 = self.t3_gate(z)

        return {
            "t1":  self.h1(z),
            "t2":  self.h2(z),
            "t3t": self.h3t(z_t3),
            "t3s": self.h3s(z_t3),
            "t4":  self.h4(z)
        }
