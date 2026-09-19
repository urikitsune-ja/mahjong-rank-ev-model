# rank-ev-model: 局開始時順位予測 & 段位PT EVモデル (公開版)

本リポジトリは、麻雀の局開始時における点数および局情報から、最終順位（1〜4位）の確率分布を予測し、その確率分布から天鳳7段・9段相当の段位PT平均収支（EV）を算出するモデル・パッケージの公開版実装です。

---

## 主な特徴

- **自己完結型オープンソース**: 実対局データやプロプライエタリなデータ変換コードを含まず、動作確認・学習用の完全合成データ（ダミーデータ）および学習・評価・推論コード一式を提供します。
- **2段階確率予測**: 局終了確率モデル（EndPredictor）と順位分布予測モデル（RankDistributionNN）によるStacking構成。
- **確率校正 (Calibration)**: Isotonic RegressionおよびSinkhorn-Knopp正規化による高品質な確率予測。
- **Web UI & ローカル推論API**: FastAPIサーバーとレスポンシブWeb UIを内包し、ローカル環境で簡単に推論とEV計算を試せます。

---

## ディレクトリ構成

```text
rank-ev-model/
├── data/                       # 完全合成データおよび分割マスタ (game_to_fold.json)
├── models/                     # モデルファイル
├── examples/                   # サンプルデータ生成スクリプト
│   └── generate_dummy_data.py
├── src/                        # コア実装
│   ├── dataset.py              # 特徴量抽出・前処理
│   ├── models.py               # PyTorch NN モデル定義
│   ├── train.py                # 学習・10-Fold CV・校正パイプライン
│   ├── eval.py                 # 評価指標計算 (LogLoss, ROC-AUC, ECE, Brier Score)
│   ├── ev_calculator.py        # PT EV計算 & Sinkhorn-Knopp 正規化
│   └── server.py               # FastAPI 推論サーバー & HTML 配信
├── web/                        # Web クライアント
│   └── index.html              # 入力フォーム & 予測結果表示 UI
├── tests/                      # パイプライン自動テスト
│   └── test_pipeline.py
├── requirements.txt            # 依存ライブラリ一覧
├── LICENSE                     # MIT License
└── README.md
```

---

## インストール & セットアップ

### 依存ライブラリのインストール

Linux / macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows（PowerShell）:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### uv を使用する場合（任意）

uvを導入済みの場合は、次の手順でも依存ライブラリをインストールできます。

Linux / macOS:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows（PowerShell）:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

以後、作業を再開する際は仮想環境を有効化してからコマンドを実行してください。

```bash
source .venv/bin/activate
```

---

## INPUTフォーマット仕様

ローカル推論サーバー（`/api/predict_rank_manual`）およびデータセット入力仕様は以下の通りです (`multipart/form-data`または特徴量ベクトル)。

| 項目 | 型 | 説明 |
| :--- | :--- | :--- |
| `round_name` | string | 局(`東一`〜`西四`または`東1`〜`西4`) |
| `honba` | integer | 本場(0, 1, 2...) |
| `kyotaku` | integer | 供託(0, 1, 2...) |
| `score_e` | integer | 東家(Player 0)の点数(100点単位。例:25000点→`250`) |
| `score_s` | integer | 南家(Player 1)の点数(100点単位。例:25000点→`250`) |
| `score_w` | integer | 西家(Player 2)の点数(100点単位。例:25000点→`250`) |
| `score_n` | integer | 北家(Player 3)の点数(100点単位。例:25000点→`250`) |

※ 入力された点数(`score_e`〜`score_n`)および出力結果`results`の各行は、東二局以降の親番位置に関わらず起家基準の固定プレイヤー順(Player 0=東家, 1=南家, 2=西家, 3=北家)として扱われます。親番特徴量(`is_oya`)は指定された局名(`round_name`)から自動計算して付与されます。

### 同梱モデルについて

`models/`ディレクトリには、天鳳の牌譜をもとに学習した推論用成果物を同梱しています。

- `end_predictor.pth`:局終了確率モデルの重み
- `rank_distribution.pth`:最終順位分布モデルの重み
- `rank_calibrator.joblib`:順位確率の校正器

これらの成果物が存在する場合、追加の学習を行わずに推論サーバーを起動できます。
なお、モデル出力は推定値です。

### データ利用上の注意 & 完全合成サンプルデータについて

- 本公開版に含まれるサンプルデータ（`examples/generate_dummy_data.py`で生成されるデータ）は、天鳳等の実対局データ由来ではなく、アルゴリズムによって生成された完全な合成データ（擬似データ）です。牌譜そのもの、牌譜の再配布物、および個人を特定できる情報は本リポジトリに含まれていません。
- ユーザーご自身で取得・利用権限を確認した対局ログ（じゃんたまや天鳳など）をもとに本モデルフォーマットを作成し、自己責任で学習・利用することができます。
- サンプルデータの生成や学習時にはリポジトリ内のdata/featuresとmodelsを直接上書きするので、実行時はご注意ください。
---

## 使い方

### 1. サンプルデータの生成

```bash
python examples/generate_dummy_data.py
```

`data/features/rank_dist_dataset_00000001.npz` 〜 `rank_dist_dataset_00000100.npz` が生成されます。

### 2. モデルおよび校正器の学習・評価

```bash
python src/train.py --epochs 30
```

- **Phase 1**: 全10-Foldによる局終了確率（EndPredictor）のOOF予測を生成
- **Phase 2**: 完成したOOF局終了確率を17次元目として追加し、全10-Foldで順位分布モデル（RankDistributionNN）のOOF予測を生成
- **Phase 3**: OOF順位予測確率に対するIsotonic Regression校正器の学習、局面単位Sinkhorn-Knopp正規化の適用、評価指標の計算および評価成果物（`metrics.json`, `metrics.md`,PNG画像）の自動保存
- **Phase 4**: 全データを用いた最終モデルおよび校正器の保存 (`models/end_predictor.pth`, `models/rank_distribution.pth`, `models/rank_calibrator.joblib`)

---

## 評価指標・成果物および注意点

学習スクリプト (`src/train.py`) を実行すると、既定で`models/evaluation/`ディレクトリ配下に以下の評価成果物が保存されます。

### 出力成果物一覧
- **`metrics.json`**: 実行日時・ハイパーパラメータ・モデル構造・分割方式・全詳細評価指標を含む構造化 JSON
- **`metrics.md`**: Markdown 形式の評価レポート（注意事項、実行メタデータ、指標比較表）
- **`end_predictor_calibration.png`**: 局終了予測モデルの校正曲線プロット (Reliability Diagram)
- **`rank_distribution_calibration.png`**: 順位分布予測モデルの 3 段階比較 Reliability Diagram (2x2 サブプロット)

---

###  評価に関する重要な注意点

> [!IMPORTANT]
> 1. **チャンク単位10-Fold OOF評価**: 本評価はデータセットのチャンク単位による10-Fold Cross Validation(Out-Of-Fold: OOF)の結果です。
> 2. **対局リークの防止**: 同一対局が複数チャンクへ分割されない前提に基づき、対局単位のデータリークを防止しています。
> 3. **時系列評価との違い**: 日付単位や期間単位による時系列汎化性能の評価ではありません。
> 4. **EndPredictor教師データの自動導出**: EndPredictorの教師データ（`y_end`）は、各対局内で局進行度（局・本場）が最大の局面を終局局面として自動導出しています。
> 5. **過大解釈の防止**: EndPredictorの評価値を独立した局終了イベント予測の性能として過大解釈しないでください。
> 6. **Sinkhorn前後の明記**: 順位分布モデルのOOF評価には、等張回帰＋行正規化のみを適用した『Sinkhorn前』と、局面単位の4人組に対しSinkhorn-Knopp交互スケーリング正規化を適用した「Sinkhorn後（最終API推論と同等）」の両方を掲載しています。

---

### 評価指標の定義

- **LogLoss**: 予測確率と真のラベル間の交差エントロピー損失（多クラスLogLoss/二値LogLoss）。
- **ROC-AUC**: 局終了予測における識別能力を表す曲線下面積。
- **Top-1 Accuracy**: 最も確率が高い順位予測が実際に正解となった割合（正解率）。
- **Brier Score**: 確率予測におけるMean Squared Error(MSE)相当の指標。各順位 $k$ の二値指標 $y_{k} \in \{0, 1\}$ と予測確率 $p_k$ に対し $(p_k - y_k)^2$ の平均を計算。
- **ECE(Expected Calibration Error)**: 予測確率を10個のbinに分割し、各bin内の平均予測確率と実測発生率の絶対誤差を加重平均した確率校正誤差。

---

### 確率校正の3段階比較定義

1. **Raw(校正前)**: ニューラルネットワーク（RankDistributionNN）の生の出力LogitにSoftmaxを適用した確率。
2. **Calibrated(Sinkhorn前)**: クラス別Isotonic Regression（等張回帰）を適用し、確率の行和が1となるよう行正規化を行った確率。
3. **Calibrated+Sinkhorn(Sinkhorn後/最終API推論と同等)**: 局面単位の4人組（4行4列の確率行列）に対し、各プレイヤーの行和=1および各順位の列和=1を満たすようSinkhorn-Knopp交互スケーリング正規化を適用した、最終API推論と同等の出力確率。

---

### 合成サンプルデータにおける評価値例(デフォルト設定)

| モデル / 評価項目 | 評価指標 | 校正前(Raw) | 校正後(Sinkhorn前) | 校正後(Sinkhorn後/最終API同等) |
| :--- | :--- | :--- | :--- | :--- |
| **EndPredictor** | LogLoss / ECE | - | - | LogLoss: `0.612` / ECE: `0.193` |
| **RankDistributionNN** | 多クラス LogLoss | `1.3850` | `1.3739` | **`1.3690`** |
| **RankDistributionNN** | Top-1 Accuracy | `26.85%` | `29.00%` | **`32.77%`** |
| **RankDistributionNN** | Brier Score 平均 | `0.1873` | `0.1860` | **`0.1854`** |
| **RankDistributionNN** | ECE 平均 | `0.0058` | `0.0047` | **`0.0036`** |

---

### 3. 推論サーバーの起動

同梱済みモデルを用いて、追加の学習なしで推論を実行できます。

```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

### 4. ブラウザで Web UI を開く

ブラウザで以下の URL にアクセスします:
[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

入力フォームに局情報と各プレイヤーの点数を入力し、「順位確率 & EVを予測」ボタンをクリックすると、1〜4位確率および天鳳7段・9段の期待値（EV）が表示されます。

---

## テストの実行

```bash
python tests/test_pipeline.py
```

---

## ライセンス

本リポジトリのコード、Web UI、および合成サンプルデータは [MIT License](LICENSE) のもとで公開されています。
