# -*- coding: utf-8 -*-
"""
PT EV (期待値) 計算および Sinkhorn-Knopp による確率分布の正規化モジュール。
"""

import numpy as np
from typing import List, Dict, Union

def sinkhorn_knopp(probs: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    4人×4順位の予測確率行列 [4, 4] に対し、
    - 各行の和 = 1 (各プレイヤーの順位確率の和が1)
    - 各列の和 = 1 (各順位になるプレイヤー数の和が1)
    を満たすよう Sinkhorn-Knopp アルゴリズムで交互スケーリング正規化を行う。
    """
    P = np.copy(probs).astype(np.float64)
    # ゼロ防止のための微小値
    P = np.clip(P, 1e-12, None)
    
    for _ in range(max_iter):
        # 行正規化
        row_sums = P.sum(axis=1, keepdims=True)
        P /= row_sums
        
        # 列正規化
        col_sums = P.sum(axis=0, keepdims=True)
        P /= col_sums
        
        # 収束チェック (行和・列和が1に近いか)
        if np.max(np.abs(P.sum(axis=1) - 1.0)) < tol and np.max(np.abs(P.sum(axis=0) - 1.0)) < tol:
            break
            
    # 最終的な行正規化
    P /= P.sum(axis=1, keepdims=True)
    return P

def calculate_pt_ev(probs_1to4: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    1〜4位の確率 [p1, p2, p3, p4] から、天鳳7段および9段相当のPT EVを計算する。
    - 天鳳7段相当: EV_7 = 90 * p1 + 45 * p2 + 0 * p3 - 135 * p4
    - 天鳳9段相当: EV_9 = 90 * p1 + 45 * p2 + 0 * p3 - 165 * p4
    """
    p = np.array(probs_1to4, dtype=np.float64)
    if len(p) != 4:
        raise ValueError(f"Probability array must have length 4, got {len(p)}")
        
    ev_7 = float(90.0 * p[0] + 45.0 * p[1] + 0.0 * p[2] - 135.0 * p[3])
    ev_9 = float(90.0 * p[0] + 45.0 * p[1] + 0.0 * p[2] - 165.0 * p[3])
    
    return {
        "ev_7": ev_7,
        "ev_9": ev_9
    }
