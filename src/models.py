# -*- coding: utf-8 -*-
"""
PyTorch モデルおよび Calibration (確率校正器) 定義モジュール。

1. EndPredictorNN: 16次元特徴量から今局終了確率 (0〜1) を予測するモデル (Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid)。
2. RankDistributionNN: 17次元特徴量 (16次元 + P_end) から最終順位分布 (1〜4位) を予測するモデル (Linear -> ReLU -> Linear -> ReLU -> Linear)。
3. MultiClassIsotonicCalibrator: Isotonic Regression によるクラス別確率校正器。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

class EndPredictorNN(nn.Module):
    """
    今局終了確率予測ネットワーク (16d -> 1d Sigmoid)
    """
    def __init__(self, in_features: int = 16):
        super(EndPredictorNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RankDistributionNN(nn.Module):
    """
    順位分布予測ネットワーク (17d -> 4d Logits)
    """
    def __init__(self, in_features: int = 17, arch: str = "128-64"):
        super(RankDistributionNN, self).__init__()
        if arch == "128-64":
            self.net = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4)
            )
        elif arch == "256-256-128":
            self.net = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 4)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 4)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiClassIsotonicCalibrator:
    """
    各クラス (1〜4位) ごとに Isotonic Regression を適用する校正器クラス。
    """
    def __init__(self):
        self.calibrators = [IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0) for _ in range(4)]

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        for r in range(4):
            y_binary = (y_true == r).astype(float)
            self.calibrators[r].fit(probs[:, r], y_binary)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probs)
        for r in range(4):
            calibrated[:, r] = self.calibrators[r].predict(probs[:, r])
        # 行正規化
        sums = np.sum(calibrated, axis=1, keepdims=True)
        sums = np.where(sums == 0, 1.0, sums)
        return calibrated / sums
