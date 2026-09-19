# -*- coding: utf-8 -*-
"""
ローカル推論サーバー (FastAPI)。
FastAPI アプリケーションから推論APIおよびHTML UIを配信します。
"""

import os
import sys
import numpy as np
import torch
import joblib
from fastapi import FastAPI, Form, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any

# 同一ディレクトリおよびリポジトリルートの参照を確保
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import build_16d_features
from models import EndPredictorNN, RankDistributionNN, MultiClassIsotonicCalibrator
from ev_calculator import sinkhorn_knopp, calculate_pt_ev

app = FastAPI(title="Rank Distribution & EV Prediction Server")

device = torch.device("cpu")

# パス設定
MODEL_DIR = os.path.join(BASE_DIR, "models")
END_MODEL_PATH = os.path.join(MODEL_DIR, "end_predictor.pth")
RANK_MODEL_PATH = os.path.join(MODEL_DIR, "rank_distribution.pth")
CALIB_PATH = os.path.join(MODEL_DIR, "rank_calibrator.joblib")
WEB_HTML_PATH = os.path.join(BASE_DIR, "web", "index.html")

end_model = None
rank_model = None
rank_calibrator = None

def load_models_if_needed() -> bool:
    global end_model, rank_model, rank_calibrator
    if end_model is not None and rank_model is not None and rank_calibrator is not None:
        return True

    if not (os.path.exists(END_MODEL_PATH) and os.path.exists(RANK_MODEL_PATH) and os.path.exists(CALIB_PATH)):
        return False

    try:
        e_model = EndPredictorNN(in_features=16).to(device)
        e_model.load_state_dict(torch.load(END_MODEL_PATH, map_location=device))
        e_model.eval()

        r_model = RankDistributionNN(in_features=17).to(device)
        r_model.load_state_dict(torch.load(RANK_MODEL_PATH, map_location=device))
        r_model.eval()

        calib = joblib.load(CALIB_PATH)

        end_model = e_model
        rank_model = r_model
        rank_calibrator = calib
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load models or calibrator: {e}")
        return False

@app.get("/api/health")
async def health_check():
    """
    サーバーおよび必要な学習済み成果物（モデル重み・校正器）が推論可能な状態かを返す。
    """
    models_ready = load_models_if_needed()
    if not models_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Required model weights or rank calibrator not found or failed to load. Please run train.py to generate models and calibrator first."
            }
        )
    return {"status": "ok", "message": "Server, models, and calibrator are ready."}

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """
    HTML UI を配信する。
    """
    if os.path.exists(WEB_HTML_PATH):
        with open(WEB_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>web/index.html not found.</h1>", status_code=404)

@app.post("/api/predict_rank_manual")
async def predict_rank_manual(
    round_name: str = Form(...),
    honba: int = Form(...),
    kyotaku: int = Form(...),
    score_e: int = Form(...),
    score_s: int = Form(...),
    score_w: int = Form(...),
    score_n: int = Form(...)
):
    if not load_models_if_needed():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required model weights or rank calibrator not found. Please train models and calibrator first."
        )

    round_map = {
        "東一": 1.0, "東二": 2.0, "東三": 3.0, "東四": 4.0,
        "南一": 5.0, "南二": 6.0, "南三": 7.0, "南四": 8.0,
        "西一": 9.0, "西二": 10.0, "西三": 11.0, "西四": 12.0,
        "東1": 1.0, "東2": 2.0, "東3": 3.0, "東4": 4.0,
        "南1": 5.0, "南2": 6.0, "南3": 7.0, "南4": 8.0,
        "西1": 9.0, "西2": 10.0, "西3": 11.0, "西4": 12.0,
    }
    # 未定義の round_name は東一 (1.0) として扱う
    round_val = round_map.get(round_name, 1.0)
    scores_rounded = [
        float(score_e * 100),
        float(score_s * 100),
        float(score_w * 100),
        float(score_n * 100)
    ]

    feats = []
    for p in range(4):
        feat = build_16d_features(
            round_val=round_val,
            honba=float(honba),
            kyotaku=float(kyotaku),
            scores=scores_rounded,
            p=p
        )
        feats.append(feat)

    feats_np = np.array(feats, dtype=np.float32)

    with torch.no_grad():
        feats_tensor = torch.from_numpy(feats_np).to(device)
        p_end = end_model(feats_tensor)
        feats_17 = torch.cat([feats_tensor, p_end], dim=1)
        logits = rank_model(feats_17)
        raw_probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Isotonic Regression 校正の適用
    calibrated_probs = rank_calibrator.transform(raw_probs)
    final_probs = sinkhorn_knopp(calibrated_probs)

    results = []
    for p in range(4):
        p_probs = final_probs[p]
        ev_dict = calculate_pt_ev(p_probs)

        results.append({
            "seat": p,
            "score": int(scores_rounded[p]),
            "probs": [float(p_probs[r]) for r in range(4)],
            "ev_7": float(ev_dict["ev_7"]),
            "ev_9": float(ev_dict["ev_9"])
        })

    return {
        "round_name": round_name,
        "round_val": round_val,
        "honba": honba,
        "kyotaku": kyotaku,
        "results": results
    }
