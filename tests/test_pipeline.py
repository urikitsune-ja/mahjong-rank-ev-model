# -*- coding: utf-8 -*-
"""
統合パイプラインおよび API エンドポイントの自動テスト (unittest)。
"""

import os
import sys
import unittest
import numpy as np

# インポートパスの設定
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ev_calculator import calculate_pt_ev, sinkhorn_knopp
from examples.generate_dummy_data import generate_dummy_dataset
from train import main as train_main

class TestPipeline(unittest.TestCase):

    def test_ev_calculator(self):
        probs = [0.4, 0.3, 0.2, 0.1]
        res = calculate_pt_ev(probs)
        # EV7 = 90*0.4 + 45*0.3 + 0*0.2 - 135*0.1 = 36 + 13.5 + 0 - 13.5 = 36.0
        # EV9 = 90*0.4 + 45*0.3 + 0*0.2 - 165*0.1 = 36 + 13.5 + 0 - 16.5 = 33.0
        self.assertAlmostEqual(res["ev_7"], 36.0, places=4)
        self.assertAlmostEqual(res["ev_9"], 33.0, places=4)

    def test_sinkhorn_knopp(self):
        probs = np.array([
            [0.4, 0.3, 0.2, 0.1],
            [0.3, 0.3, 0.2, 0.2],
            [0.2, 0.2, 0.4, 0.2],
            [0.1, 0.2, 0.2, 0.5]
        ])
        res = sinkhorn_knopp(probs)
        # 行和と列和がすべて 1.0 に近いか
        np.testing.assert_allclose(res.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(res.sum(axis=0), 1.0, atol=1e-5)

    def test_dummy_data_and_training(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data", "features")
        models_dir = os.path.join(project_root, "models")

        # 1. 完全合成ダミーデータ生成 (100チャンク, 各10局面)
        generate_dummy_dataset(data_dir, num_chunks=100, situations_per_chunk=10)
        self.assertTrue(os.path.exists(os.path.join(data_dir, "rank_dist_dataset_00000001.npz")))
        self.assertTrue(os.path.exists(os.path.join(data_dir, "rank_dist_dataset_00000100.npz")))

        # 2. モデルおよび校正器の学習実行 (Phase 1 〜 Phase 4)
        sys.argv = ["train.py", "--data_dir", data_dir, "--out_dir", models_dir, "--epochs", "2"]
        train_main()

        # 3. 必須成果物の生成確認
        self.assertTrue(os.path.exists(os.path.join(models_dir, "end_predictor.pth")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "rank_distribution.pth")))
        self.assertTrue(os.path.exists(os.path.join(models_dir, "rank_calibrator.joblib")))

        # 4. 評価成果物 (metrics.json, metrics.md, 校正曲線PNG) の生成確認
        eval_dir = os.path.join(models_dir, "evaluation")
        json_path = os.path.join(eval_dir, "metrics.json")
        md_path = os.path.join(eval_dir, "metrics.md")
        end_png_path = os.path.join(eval_dir, "end_predictor_calibration.png")
        rank_png_path = os.path.join(eval_dir, "rank_distribution_calibration.png")

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))
        self.assertTrue(os.path.exists(end_png_path))
        self.assertTrue(os.path.exists(rank_png_path))

        # 5. metrics.json / metrics.md の内容構造アサーション
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            m_json = json.load(f)

        self.assertIn("metadata", m_json)
        self.assertIn("end_predictor_oof", m_json)
        self.assertIn("rank_distribution_oof", m_json)
        self.assertIn("raw", m_json["rank_distribution_oof"])
        self.assertIn("calibrated", m_json["rank_distribution_oof"])
        self.assertIn("calibrated_sinkhorn", m_json["rank_distribution_oof"])
        self.assertIn("log_loss", m_json["end_predictor_oof"])
        self.assertIn("positive_rate", m_json["end_predictor_oof"])
        self.assertIn("top1_accuracy", m_json["rank_distribution_oof"]["calibrated_sinkhorn"])
        self.assertIn("mean_brier_score", m_json["rank_distribution_oof"]["calibrated_sinkhorn"])

        with open(md_path, "r", encoding="utf-8") as f:
            m_md = f.read()

        # 注意書きの存在確認
        self.assertIn("本評価はデータセットのチャンク単位による 10-Fold Cross Validation", m_md)
        self.assertIn("同一対局が複数チャンクへ分割されない前提に基づき", m_md)
        self.assertIn("日付単位や期間単位による時系列汎化性能の評価ではありません", m_md)
        self.assertIn("EndPredictor の教師データ（`y_end`）は", m_md)
        self.assertIn("EndPredictor の評価値を独立した局終了イベント予測の性能として過大解釈しないでください", m_md)
        self.assertIn("Sinkhorn後（最終API推論と同等）", m_md)

    def test_seat_ordering_consistency(self):
        """
        Issue #28 回帰テスト:
        東二局で4人の点数を指定した際、特徴量に渡される点数が回転せず
        固定順 [Player 0..3] となっており、Player 1 (南家) にのみ is_oya=1.0 が立つことを検証する。
        """
        from dataset import build_16d_features

        round_val = 2.0  # 東二局
        honba = 0.0
        kyotaku = 0.0
        scores_rounded = [28000.0, 25000.0, 24000.0, 23000.0]

        # Player 0 (東家) の特徴量検証
        feat_p0 = build_16d_features(round_val, honba, kyotaku, scores_rounded, p=0)
        self.assertAlmostEqual(feat_p0[3], 0.28, places=4)
        self.assertAlmostEqual(feat_p0[4], 0.25, places=4)
        self.assertAlmostEqual(feat_p0[5], 0.24, places=4)
        self.assertAlmostEqual(feat_p0[6], 0.23, places=4)
        self.assertEqual(feat_p0[12], 0.0)  # is_oya = 0.0 (東二局の東家は子)

        # Player 1 (南家) の特徴量検証
        feat_p1 = build_16d_features(round_val, honba, kyotaku, scores_rounded, p=1)
        self.assertAlmostEqual(feat_p1[3], 0.28, places=4)
        self.assertAlmostEqual(feat_p1[4], 0.25, places=4)
        self.assertEqual(feat_p1[12], 1.0)  # is_oya = 1.0 (東二局の南家は親)

if __name__ == "__main__":
    unittest.main()
