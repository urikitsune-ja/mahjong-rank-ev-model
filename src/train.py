# -*- coding: utf-8 -*-
"""
モデル学習および Calibration (確率校正) パイプライン。
モデル構成、パラメータ設定、および y_end (局終了教師) 自動導出ロジックを採用。
チャンクファイル名（例: rank_dist_dataset_00000001）を単位として Fold を制御し、データリークを防止します。
"""

import os
import sys
import json
import argparse
import joblib
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# パス追加
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from dataset import load_dataset
from models import EndPredictorNN, RankDistributionNN, MultiClassIsotonicCalibrator
from eval import evaluate_end_predictor, evaluate_rank_distribution, save_evaluation_artifacts, apply_sinkhorn_to_oof

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_end_predictor_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = model(X_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)

def train_rank_nn_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)

def find_game_to_fold_path(data_dir: str, custom_path: str = None) -> str:
    """
    data_dir またはその親ディレクトリから game_to_fold.json を自動探索する。
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    candidates = [
        os.path.join(data_dir, "game_to_fold.json"),
        os.path.join(os.path.dirname(data_dir), "game_to_fold.json"),
        os.path.join(os.path.dirname(os.path.dirname(data_dir)), "game_to_fold.json"),
        os.path.join(REPO_ROOT, "data", "game_to_fold.json")
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None

def resolve_fold(chunk_name: str, game_to_fold: dict, idx: int) -> int:
    """
    チャンクファイル名（例: rank_dist_dataset_00000001）キーから Fold 番号を割り当てる。
    """
    if game_to_fold and chunk_name in game_to_fold:
        return int(game_to_fold[chunk_name])

    try:
        num_str = chunk_name.split("_")[-1]
        return int(num_str) % 10
    except ValueError:
        return idx % 10

def create_round_end_targets(X: np.ndarray, source_files: list) -> np.ndarray:
    """
    y_end (局終了教師) 自動導出関数。
    同一対局内で進行度 (round_val * 100 + honba * 10) が最大の局面（半荘の最終局）に 1 を立てる。
    """
    print("[INFO] Creating round-end target labels (y_end)...")
    y_end = np.zeros(len(X), dtype=np.float32)

    progress = X[:, 0] * 100.0 + X[:, 1] * 10.0

    unique_sources = list(set(source_files))
    source_to_id = {s: i for i, s in enumerate(unique_sources)}
    game_ids = np.array([source_to_id[s] for s in source_files], dtype=np.int32)

    sort_idxs = np.argsort(game_ids)
    sorted_game_ids = game_ids[sort_idxs]
    sorted_progress = progress[sort_idxs]

    diffs = np.diff(sorted_game_ids)
    change_points = np.where(diffs != 0)[0] + 1
    start_idxs = np.concatenate([[0], change_points])
    end_idxs = np.concatenate([change_points, [len(sorted_game_ids)]])

    for s_idx, e_idx in zip(start_idxs, end_idxs):
        prog_slice = sorted_progress[s_idx:e_idx]
        max_prog = np.max(prog_slice)
        local_end_idxs = np.where(prog_slice == max_prog)[0]
        y_end[sort_idxs[s_idx + local_end_idxs]] = 1.0

    num_ends = int(np.sum(y_end))
    print(f"[INFO] Successfully created y_end labels: {num_ends} end situations / {len(X)} total samples.")
    return y_end

def main():
    default_data_dir = os.path.join(REPO_ROOT, "data", "features")
    default_out_dir = os.path.join(REPO_ROOT, "models")
    default_eval_out_dir = os.path.join(REPO_ROOT, "models", "evaluation")

    parser = argparse.ArgumentParser(description="順位予測モデルの学習スクリプト")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="入力データディレクトリ")
    parser.add_argument("--game_to_fold", type=str, default=None, help="game_to_fold.json のパス (オプション)")
    parser.add_argument("--out_dir", type=str, default=default_out_dir, help="モデル保存先ディレクトリ")
    parser.add_argument("--eval_out_dir", type=str, default=default_eval_out_dir, help="評価指標・レポート出力先ディレクトリ")
    parser.add_argument("--epochs", type=int, default=30, help="エポック数 (デフォルト: 30)")
    parser.add_argument("--batch_size", type=int, default=512, help="バッチサイズ (デフォルト: 512)")
    parser.add_argument("--lr", type=float, default=0.001, help="学習率 (デフォルト: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (デフォルト: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading dataset from {args.data_dir}...")
    
    try:
        X, y, chunk_names, source_files = load_dataset(args.data_dir)
    except FileNotFoundError:
        print("[WARNING] No dataset found. Generating dummy dataset first...")
        from examples.generate_dummy_data import generate_dummy_dataset
        generate_dummy_dataset(args.data_dir, num_chunks=100, situations_per_chunk=10)
        X, y, chunk_names, source_files = load_dataset(args.data_dir)

    print(f"[INFO] Loaded {len(X)} samples from {len(set(chunk_names))} chunks.")

    # y_end (局終了教師) を自動導出
    p_end_target = create_round_end_targets(X, source_files)

    # Fold割当 (10-Fold)
    game_to_fold_path = find_game_to_fold_path(args.data_dir, args.game_to_fold)
    game_to_fold = {}
    if game_to_fold_path and os.path.exists(game_to_fold_path):
        print(f"[INFO] Loading fold master from {game_to_fold_path}")
        with open(game_to_fold_path, "r", encoding="utf-8") as f:
            game_to_fold = json.load(f)
    else:
        print("[WARNING] game_to_fold.json not found. Using chunk name index fold allocation.")

    folds = np.array([resolve_fold(c_name, game_to_fold, idx) for idx, c_name in enumerate(chunk_names)])

    oof_p_end = np.zeros((len(X), 1), dtype=np.float32)
    oof_rank_probs = np.zeros((len(X), 4), dtype=np.float32)

    # EndPredictor のエポック数 (epochs >= 30 時は 15 固定)
    end_epochs = 15 if args.epochs >= 30 else max(1, args.epochs)

    # Phase 1: EndPredictor
    print("\n" + "="*50)
    print(f" 🚀 Phase 1: EndPredictor 10-Fold Cross Validation ({end_epochs} epochs)")
    print("="*50)

    for fold in range(10):
        val_mask = (folds == fold)
        train_mask = ~val_mask

        if not np.any(val_mask) or not np.any(train_mask):
            continue

        X_tr_tensor = torch.from_numpy(X[train_mask]).float()
        y_end_tr_tensor = torch.from_numpy(p_end_target[train_mask]).float().unsqueeze(1)
        X_val_tensor = torch.from_numpy(X[val_mask]).float()

        end_dataset = TensorDataset(X_tr_tensor, y_end_tr_tensor)
        end_loader = DataLoader(end_dataset, batch_size=args.batch_size, shuffle=True)

        end_model = EndPredictorNN(in_features=16).to(device)
        optimizer = optim.Adam(end_model.parameters(), lr=args.lr)
        criterion = nn.BCELoss()

        for epoch in range(end_epochs):
            train_end_predictor_epoch(end_model, end_loader, optimizer, criterion, device)

        end_model.eval()
        with torch.no_grad():
            oof_p_end[val_mask] = end_model(X_val_tensor.to(device)).cpu().numpy()

    end_eval = evaluate_end_predictor(p_end_target, oof_p_end[:, 0])
    print(f"[PHASE 1 COMPLETE] EndPredictor OOF -> LogLoss: {end_eval['log_loss']:.4f}, AUC: {end_eval['roc_auc']:.4f}, ECE: {end_eval['ece']:.4f}")

    # Phase 2: RankDistributionNN
    print("\n" + "="*50)
    print(f" 🚀 Phase 2: RankDistributionNN 10-Fold Cross Validation ({args.epochs} epochs)")
    print("="*50)

    X17_all = np.hstack([X, oof_p_end])

    for fold in range(10):
        val_mask = (folds == fold)
        train_mask = ~val_mask

        if not np.any(val_mask) or not np.any(train_mask):
            continue

        X17_tr_tensor = torch.from_numpy(X17_all[train_mask]).float()
        y_tr_tensor = torch.from_numpy(y[train_mask]).long()

        rank_dataset = TensorDataset(X17_tr_tensor, y_tr_tensor)
        rank_loader = DataLoader(rank_dataset, batch_size=args.batch_size, shuffle=True)

        rank_model = RankDistributionNN(in_features=17).to(device)
        optimizer_rank = optim.Adam(rank_model.parameters(), lr=args.lr)
        criterion_rank = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            train_rank_nn_epoch(rank_model, rank_loader, optimizer_rank, criterion_rank, device)

        rank_model.eval()
        with torch.no_grad():
            logits = rank_model(torch.from_numpy(X17_all[val_mask]).float().to(device))
            oof_rank_probs[val_mask] = torch.softmax(logits, dim=1).cpu().numpy()

    # Phase 3: Calibration
    print("\n" + "="*50)
    print(" 📊 Phase 3: OOF Evaluation & Fitting Calibrator")
    print("="*50)

    rank_eval_raw = evaluate_rank_distribution(y, oof_rank_probs)
    print(f"RankNN Raw OOF -> Multi-class LogLoss: {rank_eval_raw['multi_class_log_loss']:.4f}")

    calibrator = MultiClassIsotonicCalibrator()
    calibrator.fit(oof_rank_probs, y)
    calibrated_oof_probs = calibrator.transform(oof_rank_probs)

    rank_eval_calib = evaluate_rank_distribution(y, calibrated_oof_probs)
    print(f"RankNN Calibrated OOF (pre-Sinkhorn) -> Multi-class LogLoss: {rank_eval_calib['multi_class_log_loss']:.4f}")

    # Sinkhorn 交互スケーリング正規化適用 (最終API推論同等)
    sinkhorn_oof_probs = apply_sinkhorn_to_oof(calibrated_oof_probs)
    rank_eval_sinkhorn = evaluate_rank_distribution(y, sinkhorn_oof_probs)
    print(f"RankNN Calibrated OOF (with Sinkhorn) -> Multi-class LogLoss: {rank_eval_sinkhorn['multi_class_log_loss']:.4f}")

    # Save Evaluation Artifacts (metrics.json, metrics.md, PNG plots)
    meta_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "epochs": args.epochs,
        "end_epochs": end_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "data_dir": os.path.abspath(args.data_dir),
        "total_samples": int(len(X)),
        "num_chunks": int(len(set(chunk_names))),
        "n_folds": 10,
        "split_method": "chunk-based 10-fold CV",
        "model_architectures": {
            "end_predictor": "Linear(16->64)->ReLU->Linear(64->32)->ReLU->Linear(32->1)->Sigmoid",
            "rank_distribution": "Linear(17->128)->ReLU->Linear(128->64)->ReLU->Linear(64->4)"
        },
        "calibration_method": "per-class isotonic regression + row normalization + Sinkhorn-Knopp at inference"
    }

    save_evaluation_artifacts(
        out_dir=args.eval_out_dir,
        meta_dict=meta_dict,
        end_eval=end_eval,
        rank_eval_raw=rank_eval_raw,
        rank_eval_calib=rank_eval_calib,
        rank_eval_sinkhorn=rank_eval_sinkhorn,
        y_end_true=p_end_target,
        oof_p_end=oof_p_end[:, 0],
        y_rank_true=y,
        oof_rank_probs=oof_rank_probs,
        calibrated_oof_probs=calibrated_oof_probs,
        sinkhorn_oof_probs=sinkhorn_oof_probs
    )

    # Phase 4: Final Models
    print("\n" + "="*50)
    print(" 💾 Phase 4: Training Final Models on Full Data")
    print("="*50)

    X_full_tensor = torch.from_numpy(X).float()
    y_end_full_tensor = torch.from_numpy(p_end_target).float().unsqueeze(1)
    full_end_loader = DataLoader(TensorDataset(X_full_tensor, y_end_full_tensor), batch_size=args.batch_size, shuffle=True)

    final_end_model = EndPredictorNN(in_features=16).to(device)
    opt_end = optim.Adam(final_end_model.parameters(), lr=args.lr)
    for _ in range(end_epochs):
        train_end_predictor_epoch(final_end_model, full_end_loader, opt_end, nn.BCELoss(), device)

    final_end_model.eval()
    with torch.no_grad():
        full_p_end = final_end_model(X_full_tensor.to(device)).cpu().numpy()

    X17_full = np.hstack([X, full_p_end])
    X17_full_tensor = torch.from_numpy(X17_full).float()
    y_full_tensor = torch.from_numpy(y).long()
    full_rank_loader = DataLoader(TensorDataset(X17_full_tensor, y_full_tensor), batch_size=args.batch_size, shuffle=True)

    final_rank_model = RankDistributionNN(in_features=17).to(device)
    opt_rank = optim.Adam(final_rank_model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        train_rank_nn_epoch(final_rank_model, full_rank_loader, opt_rank, nn.CrossEntropyLoss(), device)

    end_path = os.path.join(args.out_dir, "end_predictor.pth")
    rank_path = os.path.join(args.out_dir, "rank_distribution.pth")
    calib_path = os.path.join(args.out_dir, "rank_calibrator.joblib")

    torch.save(final_end_model.state_dict(), end_path)
    torch.save(final_rank_model.state_dict(), rank_path)
    joblib.dump(calibrator, calib_path)

    print(f"[SUCCESS] Saved final EndPredictor model to {end_path}")
    print(f"[SUCCESS] Saved final RankDistribution model to {rank_path}")
    print(f"[SUCCESS] Saved RankCalibrator to {calib_path}")

if __name__ == "__main__":
    main()
