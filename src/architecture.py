"""
src/architecture.py

TriGuard-DF v1 model architecture.
Copied verbatim from 02_train.py to ensure weight compatibility.
"""

import torch
import torch.nn as nn
import timm


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        w = torch.softmax(self.scorer(x), dim=1)
        return (x * w).sum(dim=1)


class SpatialBranch(nn.Module):
    BACKBONE_DIM = 1792

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False,
            num_classes=0, global_pool="avg",
        )
        self.pool = TemporalAttentionPool(self.BACKBONE_DIM)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.BACKBONE_DIM),
            nn.Dropout(dropout),
            nn.Linear(self.BACKBONE_DIM, embed_dim),
            nn.GELU(),
        )

    def forward(self, faces):
        B, T, C, H, W = faces.shape
        x = self.backbone(faces.view(B * T, C, H, W))
        x = x.view(B, T, -1)
        return self.proj(self.pool(x))


class ResBlock2d(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class SpectralBranch(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(), ResBlock2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(), ResBlock2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(), ResBlock2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.GELU(),
        )

    def forward(self, fft):
        B, T, H, W = fft.shape
        x = self.cnn(fft.view(B * T, 1, H, W))
        x = self.proj(x)
        return x.view(B, T, -1).mean(dim=1)


class PhysioBranch(nn.Module):
    def __init__(self, n_rois: int, clip_f: int,
                 embed_dim: int, dropout: float):
        super().__init__()
        roi_d = 128

        self.roi_proj = nn.Linear(clip_f, roi_d)
        self.roi_pos = nn.Parameter(torch.randn(1, n_rois, roi_d) * 0.02)
        enc_a = nn.TransformerEncoderLayer(
            d_model=roi_d, nhead=4, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.roi_tr = nn.TransformerEncoder(enc_a, num_layers=2)
        self.roi_out = nn.Sequential(nn.Linear(roi_d, 64), nn.GELU())

        self.coh_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_rois * n_rois, 128), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(),
        )

        self.merge = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, rppg, coh):
        x = self.roi_proj(rppg) + self.roi_pos
        x = self.roi_out(self.roi_tr(x)).mean(1)
        y = self.coh_mlp(coh)
        return self.merge(torch.cat([x, y], dim=-1))


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, s, f, p):
        B = s.shape[0]
        tokens = torch.cat([
            self.cls.expand(B, -1, -1),
            s.unsqueeze(1), f.unsqueeze(1), p.unsqueeze(1),
        ], dim=1)
        out = self.transformer(tokens + self.pos)
        return self.norm(out[:, 0])


class TriGuardNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D = cfg["embed_dim"]
        drop = cfg["dropout"]

        self.spatial = SpatialBranch(D, drop)
        self.spectral = SpectralBranch(D, drop)
        self.physio = PhysioBranch(
            cfg["n_rois"], cfg["clip_f"], D, drop
        )
        self.fusion = CrossModalFusion(
            D, cfg["n_heads"], cfg["n_fusion_layers"], drop
        )
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(D, D // 2), nn.GELU(),
            nn.Linear(D // 2, 1),
        )

    def forward(self, faces, fft, rppg, coh):
        s = self.spatial(faces)
        f = self.spectral(fft)
        p = self.physio(rppg, coh)
        z = self.fusion(s, f, p)
        return self.head(z).squeeze(-1)