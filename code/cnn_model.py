"""Scaled-down LightX3ECG architecture (Le et al., Biomed. Signal Process.
Control 85:104963, 2023), reimplemented standalone from the paper and its
published code (not imported, per the untrusted-external-code rule) for a
3-lead, 125 Hz, base_channels=16 CPU-feasible variant -- see
papers/cpu-ecg-crosshospital/README.md for the feasibility measurements
that picked these sizes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DSConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                  padding=padding, stride=stride, groups=in_channels, bias=False)
        self.pw_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class LightSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.s_conv = DSConv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.act_fn = nn.ReLU()
        self.e_conv = DSConv1d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        a = self.pool(x)
        a = self.act_fn(self.s_conv(a))
        a = self.e_conv(a)
        return x * torch.sigmoid(a)


class LightSEResBlock(nn.Module):
    def __init__(self, in_channels, downsample=False):
        super().__init__()
        if downsample:
            self.out_channels = in_channels * 2
            self.conv_1 = DSConv1d(in_channels, self.out_channels, kernel_size=7, padding=3, stride=2)
            self.identity = nn.Sequential(
                DSConv1d(in_channels, self.out_channels, kernel_size=1, padding=0, stride=2),
                nn.BatchNorm1d(self.out_channels),
            )
        else:
            self.out_channels = in_channels
            self.conv_1 = DSConv1d(in_channels, self.out_channels, kernel_size=7, padding=3, stride=1)
            self.identity = nn.Identity()
        self.conv_2 = DSConv1d(self.out_channels, self.out_channels, kernel_size=7, padding=3, stride=1)
        self.convs = nn.Sequential(
            self.conv_1, nn.BatchNorm1d(self.out_channels), nn.ReLU(), nn.Dropout(0.3),
            self.conv_2, nn.BatchNorm1d(self.out_channels), LightSEModule(self.out_channels),
        )
        self.act_fn = nn.ReLU()

    def forward(self, x):
        return self.act_fn(self.convs(x) + self.identity(x))


class LightSEResNet18(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        b = LightSEResBlock
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=15, padding=7, stride=2),
            nn.BatchNorm1d(base_channels), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, padding=1, stride=2),
        )
        self.stage_0 = nn.Sequential(b(base_channels), b(base_channels))
        self.stage_1 = nn.Sequential(b(base_channels * 1, downsample=True), b(base_channels * 2))
        self.stage_2 = nn.Sequential(b(base_channels * 2, downsample=True), b(base_channels * 4))
        self.stage_3 = nn.Sequential(b(base_channels * 4, downsample=True), b(base_channels * 8))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage_0(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        return self.pool(x)


class LightX3ECG(nn.Module):
    """3-lead model: one LightSEResNet18 backbone per lead, fused by a learned
    attention over the three backbones' pooled features."""

    def __init__(self, base_channels=16, num_classes=5):
        super().__init__()
        self.backbone_0 = LightSEResNet18(base_channels)
        self.backbone_1 = LightSEResNet18(base_channels)
        self.backbone_2 = LightSEResNet18(base_channels)
        self.lw_attention = nn.Sequential(
            nn.Linear(base_channels * 24, base_channels * 8), nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(base_channels * 8, 3),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(base_channels * 8, num_classes))

    def forward(self, x):
        f0 = self.backbone_0(x[:, 0, :].unsqueeze(1)).squeeze(2)
        f1 = self.backbone_1(x[:, 1, :].unsqueeze(1)).squeeze(2)
        f2 = self.backbone_2(x[:, 2, :].unsqueeze(1)).squeeze(2)
        scores = torch.sigmoid(self.lw_attention(torch.cat([f0, f1, f2], dim=1)))
        merged = torch.sum(torch.stack([f0, f1, f2], dim=1) * scores.unsqueeze(-1), dim=1)
        return self.classifier(merged)
