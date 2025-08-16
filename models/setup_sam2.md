# SAM2セットアップガイド

## 1. 環境準備

### 必要なパッケージのインストール

```bash
# Python環境の作成（推奨）
python -m venv sam2_env
source sam2_env/bin/activate  # Mac/Linux
# or
sam2_env\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install torch torchvision  # PyTorch
pip install opencv-python matplotlib numpy
pip install git+https://github.com/facebookresearch/sam2.git
```

## 2. モデルのダウンロード

```bash
# ダウンロードスクリプトを実行
cd models
chmod +x download_sam2.sh
./download_sam2.sh

# または直接ダウンロード
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

### 利用可能なSAM2.1モデル

| モデル | パラメータ数 | 推奨用途 | ダウンロードサイズ |
|--------|------------|---------|-----------------|
| Tiny | 39M | モバイル・軽量処理 | ~160MB |
| Small | 46M | バランス型 | ~185MB |
| Base+ | 81M | 高精度・標準 | ~320MB |
| Large | 224M | 最高精度 | ~900MB |

## 3. SAM2の実行

### インタラクティブモード（推奨）

```bash
python test_sam2.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image path/to/your/image.jpg \
    --model-cfg sam2_hiera_b+ \
    --mode interactive
```

画像が表示されたら：
- **左クリック**: ポジティブポイント（含める領域）
- **右クリック**: ネガティブポイント（除外する領域）
- **中クリック**: 終了

### バッチモード

```bash
python test_sam2.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image path/to/your/image.jpg \
    --mode batch
```

### 自動セグメンテーション

```bash
python test_sam2.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image path/to/your/image.jpg \
    --mode auto
```

### コマンドラインでポイント指定

```bash
python test_sam2.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image path/to/your/image.jpg \
    --points "100,200,1;300,400,0" \
    --box "50,50,400,400"
```

## 4. Core MLへの変換（iOS用）

```bash
cd models/converters

# SAM2.1をCore MLに変換
python convert_sam2_to_coreml.py \
    --checkpoint ../checkpoints/sam2.1_hiera_base_plus.pt \
    --model-type sam2_base_plus \
    --output ../manifests/
```

## 5. ビデオセグメンテーション（SAM2の新機能）

SAM2はビデオトラッキングにも対応しています：

```python
from sam2.video_predictor import SAM2VideoPredictor

# ビデオ用のpredictor作成
video_predictor = SAM2VideoPredictor(sam2)

# フレームごとの処理
for frame_idx, frame in enumerate(video_frames):
    if frame_idx == 0:
        # 最初のフレームでオブジェクトを指定
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=initial_points,
            labels=initial_labels,
        )
    
    # 後続フレームの予測
    out_obj_ids, out_mask_logits = video_predictor.propagate_in_video(
        inference_state=inference_state
    )
```

## 6. パフォーマンス最適化

### GPU使用（CUDA）
```bash
python test_sam2.py --device cuda ...
```

### Apple Silicon（M1/M2）
```bash
python test_sam2.py --device mps ...
```

### CPU使用
```bash
python test_sam2.py --device cpu ...
```

## 7. トラブルシューティング

### メモリ不足の場合
- より小さいモデル（Tiny/Small）を使用
- 画像サイズを縮小
- バッチサイズを減らす

### 速度が遅い場合
- GPUを使用
- MobileSAMを検討
- 画像解像度を下げる（512x512など）

### インストールエラー
```bash
# PyTorchの再インストール
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

## 8. SAMKitでの使用

変換したCore MLモデルをiOSアプリで使用：

```swift
// SAM2.1モデルの読み込み
let model = try SamModelRef.bundled(.sam2_1_base)

// セッション作成
let session = try SamSession(model: model)

// 画像設定と推論
try session.setImage(image)
let result = try await session.predict(points: points)
```

## サンプル画像での実行例

```bash
# サンプル画像のダウンロード
wget https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg

# 実行
python test_sam2.py \
    --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
    --image truck.jpg \
    --mode interactive
```

これでSAM2を試す準備が整いました！