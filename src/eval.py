# -*- coding: utf-8 -*-
"""
モデル評価指標計算および成果物 (metrics.json, metrics.md, 校正曲線PNG) 出力モジュール。
- 局終了予測: LogLoss, ROC-AUC, ECE, 陽性率, サンプル数
- 順位分布予測: 多クラスLogLoss, Top-1 Accuracy, 各順位/平均 Brier Score, 各順位/平均 ECE, サンプル数
- 確率校正評価: Raw (校正前), Calibrated (Sinkhorn前), Calibrated + Sinkhorn (Sinkhorn後/最終API同等)
- 校正曲線: EndPredictor 1枚, 順位分布 (3段階比較) 2x2プロット
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from typing import Dict, Any, Tuple, Optional

# REPO_ROOT をパスに追加して ev_calculator をインポート
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ev_calculator import sinkhorn_knopp

def apply_sinkhorn_to_oof(oof_probs: np.ndarray) -> np.ndarray:
    """
    OOF 確率 (N, 4) に対して、4サンプル（1局面の4対局者）単位で Sinkhorn-Knopp 交互スケーリング正規化を適用する。
    """
    n_samples = len(oof_probs)
    if n_samples % 4 != 0:
        print("[WARNING] Sample size is not a multiple of 4. Returning un-sinkhorned probabilities.")
        return oof_probs

    sinkhorn_probs = np.zeros_like(oof_probs)
    for i in range(0, n_samples, 4):
        p_slice = oof_probs[i:i+4, :]
        sinkhorn_probs[i:i+4, :] = sinkhorn_knopp(p_slice)

    return sinkhorn_probs

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) を計算する。
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    校正曲線 (Reliability Diagram) 用の (mean_predicted_prob, fraction_of_positives) を計算する。
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_probs = []
    frac_positives = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
            
        bin_size = np.sum(mask)
        if bin_size > 0:
            mean_probs.append(float(np.mean(y_prob[mask])))
            frac_positives.append(float(np.mean(y_true[mask])))

    return np.array(mean_probs), np.array(frac_positives)

def evaluate_end_predictor(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    局終了予測モデルの評価。
    """
    labels = [0.0, 1.0] if len(np.unique(y_true)) > 1 else None
    loss = float(log_loss(y_true, y_prob, labels=labels)) if labels is not None else float(log_loss(y_true, y_prob))
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    ece = compute_ece(y_true, y_prob)
    pos_rate = float(np.mean(y_true == 1.0))
    n_samples = int(len(y_true))

    return {
        "log_loss": loss,
        "roc_auc": auc,
        "ece": ece,
        "positive_rate": pos_rate,
        "n_samples": n_samples
    }

def evaluate_rank_distribution(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """
    順位分布予測モデルの評価 (y_true: 0-3のクラスラベル, y_probs: [N, 4])
    """
    mc_log_loss = float(log_loss(y_true, y_probs, labels=[0, 1, 2, 3]))
    top1_acc = float(np.mean(np.argmax(y_probs, axis=1) == y_true))

    per_rank_brier = []
    per_rank_ece = []

    for r in range(4):
        y_binary = (y_true == r).astype(int)
        prob_r = y_probs[:, r]
        
        brier = float(brier_score_loss(y_binary, prob_r))
        ece = compute_ece(y_binary, prob_r)
        
        per_rank_brier.append(brier)
        per_rank_ece.append(ece)

    return {
        "multi_class_log_loss": mc_log_loss,
        "top1_accuracy": top1_acc,
        "brier_scores": per_rank_brier,
        "ece_scores": per_rank_ece,
        "mean_brier_score": float(np.mean(per_rank_brier)),
        "mean_ece": float(np.mean(per_rank_ece)),
        "n_samples": int(len(y_true))
    }

def plot_end_predictor_calibration(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    """
    EndPredictor の校正曲線をプロットして保存する。
    """
    mean_probs, frac_pos = compute_calibration_curve(y_true, y_prob)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(mean_probs, frac_pos, "s-", color="#2b5c8f", label="EndPredictor OOF")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual End Rate)")
    plt.title("EndPredictor Calibration Curve (Reliability Diagram)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_rank_distribution_calibration(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calib_probs: np.ndarray,
    save_path: str,
    sinkhorn_probs: Optional[np.ndarray] = None
):
    """
    順位分布予測モデルの校正前・校正後（Sinkhorn前/後）比較校正曲線をプロットして保存する (2x2)。
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    rank_names = ["1st Rank (Top)", "2nd Rank", "3rd Rank", "4th Rank (Last)"]

    for r in range(4):
        ax = axes[r // 2, r % 2]
        y_binary = (y_true == r).astype(int)

        raw_mean, raw_frac = compute_calibration_curve(y_binary, raw_probs[:, r])
        calib_mean, calib_frac = compute_calibration_curve(y_binary, calib_probs[:, r])

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(raw_mean, raw_frac, "o--", color="#d9534f", alpha=0.7, label="Raw Model")
        ax.plot(calib_mean, calib_frac, "s--", color="#0275d8", alpha=0.8, label="Calibrated (pre-Sinkhorn)")

        if sinkhorn_probs is not None:
            sk_mean, sk_frac = compute_calibration_curve(y_binary, sinkhorn_probs[:, r])
            ax.plot(sk_mean, sk_frac, "^-", color="#5cb85c", linewidth=2, label="Calibrated + Sinkhorn (Final API)")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Rank Calibration: {rank_names[r]}")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_evaluation_artifacts(
    out_dir: str,
    meta_dict: Dict[str, Any],
    end_eval: Dict[str, Any],
    rank_eval_raw: Dict[str, Any],
    rank_eval_calib: Dict[str, Any],
    rank_eval_sinkhorn: Dict[str, Any],
    y_end_true: np.ndarray,
    oof_p_end: np.ndarray,
    y_rank_true: np.ndarray,
    oof_rank_probs: np.ndarray,
    calibrated_oof_probs: np.ndarray,
    sinkhorn_oof_probs: np.ndarray
):
    """
    metrics.json, metrics.md, および校正曲線画像 (PNG) を out_dir に保存する。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Save metrics.json
    metrics_json = {
        "metadata": meta_dict,
        "end_predictor_oof": end_eval,
        "rank_distribution_oof": {
            "raw": rank_eval_raw,
            "calibrated": rank_eval_calib,
            "calibrated_sinkhorn": rank_eval_sinkhorn
        }
    }
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # 2. Save metrics.md
    md_content = f"""# 順位分布・終局予測モデル 評価レポート

> [!IMPORTANT]
> 評価に関する注意点：
> 1. 本評価はデータセットのチャンク単位による 10-Fold Cross Validation (Out-Of-Fold: OOF) の結果です。
> 2. 同一対局が複数チャンクへ分割されない前提に基づき、対局単位のデータリークを防止しています。
> 3. 日付単位や期間単位による時系列汎化性能の評価ではありません。
> 4. EndPredictor の教師データ（`y_end`）は、各対局内で局進行度（局・本場）が最大の局面を終局局面として自動導出しています。
> 5. したがって、EndPredictor の評価値を独立した局終了イベント予測の性能として過大解釈しないでください。
> 6. 順位分布モデルの OOF 評価には、等張回帰＋行正規化のみを適用した『Sinkhorn前』と、局面単位の 4 人組に対し Sinkhorn-Knopp 交互スケーリング正規化を適用した『Sinkhorn後（最終API推論と同等）』の両方を掲載しています。

## 1. 実行メタデータ

| 項目 | 設定値 / 内容 |
| :--- | :--- |
| **実行日時** | {meta_dict.get('timestamp', 'N/A')} |
| **シード値 (Seed)** | {meta_dict.get('seed', 'N/A')} |
| **エポック数 (Epochs)** | {meta_dict.get('epochs', 'N/A')} (EndPredictor: {meta_dict.get('end_epochs', 'N/A')}) |
| **バッチサイズ** | {meta_dict.get('batch_size', 'N/A')} |
| **データディレクトリ** | `{meta_dict.get('data_dir', 'N/A')}` |
| **総サンプル数** | {meta_dict.get('total_samples', 'N/A')} |
| **チャンク数** | {meta_dict.get('num_chunks', 'N/A')} |
| **Fold 数 / 分割方式** | {meta_dict.get('n_folds', 10)} Fold / `{meta_dict.get('split_method', 'chunk-based 10-fold CV')}` |
| **EndPredictor 構造** | `{meta_dict.get('model_architectures', {}).get('end_predictor', 'N/A')}` |
| **RankDistributionNN 構造** | `{meta_dict.get('model_architectures', {}).get('rank_distribution', 'N/A')}` |
| **確率校正方式** | `{meta_dict.get('calibration_method', 'N/A')}` |

## 2. 終局確率モデル (EndPredictor) OOF 評価

| 評価指標 | 測定値 |
| :--- | :--- |
| **LogLoss** | {end_eval['log_loss']:.5f} |
| **ROC-AUC** | {end_eval['roc_auc']:.5f} |
| **ECE (Expected Calibration Error)** | {end_eval['ece']:.5f} |
| **陽性率 (y_end=1 割合)** | {end_eval['positive_rate']:.5f} ({int(end_eval['positive_rate'] * end_eval['n_samples'])} / {end_eval['n_samples']}) |
| **評価サンプル数** | {end_eval['n_samples']} |

## 3. 順位分布予測モデル (RankDistributionNN) OOF 評価

| 評価指標 | 校正前 (Raw) | 校正後 (Sinkhorn前) | 校正後 (Sinkhorn後 / 最終API同等) | 最終改善幅 (vs Raw) |
| :--- | :--- | :--- | :--- | :--- |
| **多クラス LogLoss** | {rank_eval_raw['multi_class_log_loss']:.5f} | {rank_eval_calib['multi_class_log_loss']:.5f} | {rank_eval_sinkhorn['multi_class_log_loss']:.5f} | {rank_eval_sinkhorn['multi_class_log_loss'] - rank_eval_raw['multi_class_log_loss']:+.5f} |
| **Top-1 Accuracy** | {rank_eval_raw['top1_accuracy'] * 100:.2f}% | {rank_eval_calib['top1_accuracy'] * 100:.2f}% | {rank_eval_sinkhorn['top1_accuracy'] * 100:.2f}% | {(rank_eval_sinkhorn['top1_accuracy'] - rank_eval_raw['top1_accuracy']) * 100:+.2f}% |
| **Brier Score (1位)** | {rank_eval_raw['brier_scores'][0]:.5f} | {rank_eval_calib['brier_scores'][0]:.5f} | {rank_eval_sinkhorn['brier_scores'][0]:.5f} | {rank_eval_sinkhorn['brier_scores'][0] - rank_eval_raw['brier_scores'][0]:+.5f} |
| **Brier Score (2位)** | {rank_eval_raw['brier_scores'][1]:.5f} | {rank_eval_calib['brier_scores'][1]:.5f} | {rank_eval_sinkhorn['brier_scores'][1]:.5f} | {rank_eval_sinkhorn['brier_scores'][1] - rank_eval_raw['brier_scores'][1]:+.5f} |
| **Brier Score (3位)** | {rank_eval_raw['brier_scores'][2]:.5f} | {rank_eval_calib['brier_scores'][2]:.5f} | {rank_eval_sinkhorn['brier_scores'][2]:.5f} | {rank_eval_sinkhorn['brier_scores'][2] - rank_eval_raw['brier_scores'][2]:+.5f} |
| **Brier Score (4位)** | {rank_eval_raw['brier_scores'][3]:.5f} | {rank_eval_calib['brier_scores'][3]:.5f} | {rank_eval_sinkhorn['brier_scores'][3]:.5f} | {rank_eval_sinkhorn['brier_scores'][3] - rank_eval_raw['brier_scores'][3]:+.5f} |
| **Brier Score 平均** | {rank_eval_raw['mean_brier_score']:.5f} | {rank_eval_calib['mean_brier_score']:.5f} | {rank_eval_sinkhorn['mean_brier_score']:.5f} | {rank_eval_sinkhorn['mean_brier_score'] - rank_eval_raw['mean_brier_score']:+.5f} |
| **ECE (1位)** | {rank_eval_raw['ece_scores'][0]:.5f} | {rank_eval_calib['ece_scores'][0]:.5f} | {rank_eval_sinkhorn['ece_scores'][0]:.5f} | {rank_eval_sinkhorn['ece_scores'][0] - rank_eval_raw['ece_scores'][0]:+.5f} |
| **ECE (2位)** | {rank_eval_raw['ece_scores'][1]:.5f} | {rank_eval_calib['ece_scores'][1]:.5f} | {rank_eval_sinkhorn['ece_scores'][1]:.5f} | {rank_eval_sinkhorn['ece_scores'][1] - rank_eval_raw['ece_scores'][1]:+.5f} |
| **ECE (3位)** | {rank_eval_raw['ece_scores'][2]:.5f} | {rank_eval_calib['ece_scores'][2]:.5f} | {rank_eval_sinkhorn['ece_scores'][2]:.5f} | {rank_eval_sinkhorn['ece_scores'][2] - rank_eval_raw['ece_scores'][2]:+.5f} |
| **ECE (4位)** | {rank_eval_raw['ece_scores'][3]:.5f} | {rank_eval_calib['ece_scores'][3]:.5f} | {rank_eval_sinkhorn['ece_scores'][3]:.5f} | {rank_eval_sinkhorn['ece_scores'][3] - rank_eval_raw['ece_scores'][3]:+.5f} |
| **ECE 平均** | {rank_eval_raw['mean_ece']:.5f} | {rank_eval_calib['mean_ece']:.5f} | {rank_eval_sinkhorn['mean_ece']:.5f} | {rank_eval_sinkhorn['mean_ece'] - rank_eval_raw['mean_ece']:+.5f} |
| **評価サンプル数** | {rank_eval_raw['n_samples']} | {rank_eval_calib['n_samples']} | {rank_eval_sinkhorn['n_samples']} | - |

## 4. 校正曲線 (Reliability Diagram)

- **EndPredictor 校正曲線**: `end_predictor_calibration.png`
- **順位分布モデル 校正曲線 (3段階比較)**: `rank_distribution_calibration.png`
"""
    md_path = os.path.join(out_dir, "metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 3. Plot and save PNG images
    end_png_path = os.path.join(out_dir, "end_predictor_calibration.png")
    plot_end_predictor_calibration(y_end_true, oof_p_end, end_png_path)

    rank_png_path = os.path.join(out_dir, "rank_distribution_calibration.png")
    plot_rank_distribution_calibration(y_rank_true, oof_rank_probs, calibrated_oof_probs, rank_png_path, sinkhorn_probs=sinkhorn_oof_probs)

    print(f"[SUCCESS] Saved evaluation metrics to {json_path}")
    print(f"[SUCCESS] Saved markdown report to {md_path}")
    print(f"[SUCCESS] Saved calibration plots to {end_png_path} and {rank_png_path}")


