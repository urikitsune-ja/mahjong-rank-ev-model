# -*- coding: utf-8 -*-
"""
公開用完全合成データ (100 チャンクファイル) および Fold マスタ生成スクリプト。
実対局データを一切含まず、アルゴリズムによって生成された擬似データセットを出力します。
同一対局内に複数の進行局面を持たせ、最終局判定 (y_end) が正しく動作する構成とします。
"""

import os
import json
import numpy as np

def generate_dummy_dataset(out_dir: str, num_chunks: int = 100, situations_per_chunk: int = 10):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(42)
    
    game_to_fold = {}
    base_data_dir = os.path.dirname(out_dir) if os.path.basename(out_dir) == "features" else out_dir

    for c_idx in range(1, num_chunks + 1):
        chunk_name = f"rank_dist_dataset_{c_idx:08d}"
        npz_filename = f"{chunk_name}.npz"
        out_path = os.path.join(out_dir, npz_filename)

        X_list = []
        y_list = []
        record_file_indices = []

        # 1チャンクあたり2対局 (各5局)
        games_in_chunk = [f"dummy_game_{c_idx:04d}_A", f"dummy_game_{c_idx:04d}_B"]
        source_files = games_in_chunk

        for g_local_idx, game_name in enumerate(games_in_chunk):
            # 東1局 (1) から進行局数を増加
            rounds = [1, 3, 5, 7, 8] # 東1, 東3, 南1, 南3, 南4(最終)
            base_scores = np.array([25000, 25000, 25000, 25000])

            for r_val in rounds:
                round_val = float(r_val)
                honba = float(np.random.randint(0, 2))
                kyotaku = float(np.random.randint(0, 2))

                score_fluctuation = np.random.randint(-100, 100, size=4) * 100
                score_fluctuation -= int(np.mean(score_fluctuation))
                scores = base_scores + score_fluctuation
                scores = np.clip(scores, 1000, 60000)

                sorted_s = sorted([(scores[i], -i) for i in range(4)], reverse=True)
                diffs = [
                    sorted_s[0][0] - scores[0],
                    sorted_s[1][0] - scores[0],
                    sorted_s[2][0] - scores[0],
                    sorted_s[3][0] - scores[0],
                    sorted_s[0][0] - scores[1],
                    sorted_s[1][0] - scores[1]
                ]

                row_x = [round_val, honba, kyotaku, scores[0], scores[1], scores[2], scores[3]] + diffs

                noise = np.random.normal(0, 5000, size=4)
                final_scores = scores + noise
                rank_order = np.argsort(-final_scores)
                ranks = np.zeros(4, dtype=np.int32)
                for r, p_idx in enumerate(rank_order):
                    ranks[p_idx] = r + 1

                X_list.append(row_x)
                y_list.append(ranks)
                record_file_indices.append(g_local_idx)

        np.savez_compressed(
            out_path,
            X=np.array(X_list, dtype=np.float32),
            y=np.array(y_list, dtype=np.int32),
            record_file_indices=np.array(record_file_indices, dtype=np.int32),
            source_files=source_files
        )

        game_to_fold[chunk_name] = (c_idx - 1) % 10

    print(f"[SUCCESS] Generated {num_chunks} chunk files in {out_dir}.")

    json_path = os.path.join(base_data_dir, "game_to_fold.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(game_to_fold, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Saved fold master to {json_path} ({len(game_to_fold)} entries).")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_data_dir = os.path.join(project_root, "data", "features")
    generate_dummy_dataset(target_data_dir, num_chunks=100, situations_per_chunk=10)
