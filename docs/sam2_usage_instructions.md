# HuggingFace SAM2モデルの使い方

## 1. モデルファイルの準備

### 必要なファイル
以下の3つの`.mlpackage`ファイルをダウンロードします：

- `SAM2TinyImageEncoderFLOAT16.mlpackage`
- `SAM2TinyPromptEncoderFLOAT16.mlpackage`
- `SAM2TinyMaskDecoderFLOAT16.mlpackage`

### Xcodeプロジェクトに追加する手順

1. **Xcodeでプロジェクトを開く**
   ```
   open samples/ios-sample/SAMKitDemo.xcodeproj
   ```

2. **プロジェクトナビゲータでファイルを追加**
   - プロジェクトナビゲータ（左側パネル）で`SAMKitDemo`フォルダを右クリック
   - "Add Files to 'SAMKitDemo'"を選択

3. **モデルファイルを選択**
   - ダウンロードした3つの`.mlpackage`ファイルを選択
   - "Copy items if needed" にチェック
   - "Target Membership"で`SAMKitDemo`にチェック

4. **追加を確認**
   - プロジェクトナビゲータに3つのファイルが表示されることを確認

## 2. サンプルアプリでの使用

### 基本的な使い方

1. **アプリを起動**
   - Xcodeでビルド・実行
   
2. **HuggingFaceモデルを有効にする**
   - "Use HuggingFace SAM2 Models" トグルをオンにする
   - ドロップダウンから"SAM2 Tiny (HF)"を選択

3. **画像を選択**
   - "Photo Library"または"Camera"ボタンで画像を選択

4. **セグメンテーション開始**
   - "Start Segmentation"をタップ
   - 画像が表示されたら、セグメントしたい部分をタップ

### 操作方法

- **ポイント追加**: 画像をタップしてポイントを追加
- **マスク表示切り替え**: "Show Mask"トグルでマスクの表示/非表示を切り替え
- **複数マスクの選択**: 複数のマスクが生成された場合、セグメントピッカーで選択
- **リセット**: "Clear Points"でポイントとマスクをクリア

## 3. コード例

### Sam2Sessionの初期化

```swift
import SAMKit

// HuggingFaceモデルの初期化
let sam2Session = try Sam2Session(
    modelName: "SAM2Tiny",
    config: .bestAvailable
)
```

### セグメンテーション実行

```swift
// 画像を設定
try sam2Session.setImage(cgImage)

// ポイントプロンプトで推論
let points = [SamPoint(x: 100, y: 100, label: .positive)]
let result = try await sam2Session.predict(points: points)

// 結果の取得
let bestMask = result.masks.first
let score = result.scores.first
```

### 従来モデルとの違い

```swift
// 従来モデル（統合）
let session = try SamSession(model: .sam2_1_base)

// HuggingFaceモデル（分離）
let sam2Session = try Sam2Session(modelName: "SAM2Tiny")
```

## 4. パフォーマンス比較

| モデル | モデルサイズ | 推論速度 | メモリ使用量 | 精度 |
|--------|-------------|----------|-------------|------|
| SAM2 Tiny (HF) | ~50MB | 高速 | 低 | 高 |
| SAM 2.1 Base | ~320MB | 中速 | 高 | 最高 |
| MobileSAM | ~40MB | 高速 | 最低 | 中 |

## 5. トラブルシューティング

### エラー: "modelNotFound"
**原因**: モデルファイルがプロジェクトに正しく追加されていない

**解決方法**:
1. 3つの`.mlpackage`ファイルすべてがプロジェクトに追加されているか確認
2. ファイル名が正確か確認（大文字小文字を含む）
3. Target Membershipが正しく設定されているか確認

### エラー: "Memory allocation failed"
**原因**: メモリ不足

**解決方法**:
1. より小さい画像サイズを使用
2. バックグラウンドアプリを終了
3. デバイスを再起動

### 推論が遅い
**解決方法**:
1. `RuntimeConfig.bestAvailable`を使用してGPU加速を有効化
2. 画像解像度を下げる
3. より小さいモデル（Tiny）を使用

## 6. デバッグ

### モデルファイルの存在確認

```swift
private func checkModelFiles() {
    let bundle = Bundle.main
    let models = [
        "SAM2TinyImageEncoderFLOAT16",
        "SAM2TinyPromptEncoderFLOAT16", 
        "SAM2TinyMaskDecoderFLOAT16"
    ]
    
    for model in models {
        if bundle.url(forResource: model, withExtension: "mlpackage") != nil {
            print("✓ \(model) found")
        } else {
            print("✗ \(model) missing")
        }
    }
}
```

### ログ出力の有効化

```swift
// SamKitのデバッグログを有効化
SamLogger.setLevel(.debug)
```

これで、HuggingFaceのSAM2 Tinyモデルをサンプルアプリで使用できます！