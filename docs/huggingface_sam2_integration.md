# HuggingFace SAM2 Tiny モデル統合ガイド

HuggingFaceで配布されているSAM2 Tinyモデル（3つの分離されたコンポーネント）をSAMKitサンプルアプリで使用する方法を説明します。

## 必要なモデル

HuggingFaceで配布されている以下の3つのモデルファイルが必要です：

- `SAM2TinyImageEncoderFLOAT16.mlpackage`
- `SAM2TinyPromptEncoderFLOAT16.mlpackage`
- `SAM2TinyMaskDecoderFLOAT16.mlpackage`

## 1. モデルファイルの準備

### ダウンロード
```bash
# HuggingFaceからモデルをダウンロード
git lfs install
git clone https://huggingface.co/apple/SAM2TinyEncoderFLOAT16
git clone https://huggingface.co/apple/SAM2TinyPromptEncoderFLOAT16
git clone https://huggingface.co/apple/SAM2TinyMaskDecoderFLOAT16
```

### Xcodeプロジェクトへの追加
1. Xcodeで`SAMKitDemo.xcodeproj`を開く
2. プロジェクトナビゲータで右クリック → "Add Files to 'SAMKitDemo'"
3. 3つの`.mlpackage`ファイルを選択
4. "Copy items if needed" をチェック
5. Target Membershipで`SAMKitDemo`をチェック

## 2. iOS実装

### Sam2Session使用例

```swift
import SAMKit

class SAM2ViewController: UIViewController {
    private var sam2Session: Sam2Session?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupSAM2()
    }
    
    private func setupSAM2() {
        do {
            // HuggingFaceモデルの初期化
            sam2Session = try Sam2Session(
                modelName: "SAM2Tiny",
                config: .bestAvailable
            )
            print("SAM2 session initialized successfully")
        } catch {
            print("Failed to initialize SAM2: \(error)")
        }
    }
    
    private func runSegmentation(image: UIImage, point: CGPoint) {
        guard let sam2Session = sam2Session,
              let cgImage = image.cgImage else { return }
        
        Task {
            do {
                // 画像を設定
                try sam2Session.setImage(cgImage)
                
                // ポイントプロンプトで推論実行
                let points = [SamPoint(x: point.x, y: point.y, label: .positive)]
                let result = try await sam2Session.predict(points: points)
                
                // 結果の表示
                await MainActor.run {
                    displayResult(result)
                }
            } catch {
                print("Segmentation failed: \(error)")
            }
        }
    }
    
    private func displayResult(_ result: SamResult) {
        // マスクの可視化
        if let bestMask = result.masks.first {
            let maskImage = bestMask.cgImage
            // UIImageViewに表示など
        }
    }
}
```

### 従来のSamSessionとの比較

```swift
// 従来のSamSession（統合モデル）
let session = try SamSession(model: .sam2_1_base)

// HuggingFace SAM2（分離モデル）
let sam2Session = try Sam2Session(modelName: "SAM2Tiny")
```

## 3. サンプルアプリの更新

### ContentView.swiftの修正例

```swift
struct ContentView: View {
    @State private var selectedModelType: ModelType = .sam2Tiny
    @State private var session: SamSession?
    @State private var sam2Session: Sam2Session?
    
    enum ModelType: String, CaseIterable {
        case mobileSAM = "MobileSAM"
        case sam2_1_base = "SAM 2.1 Base"
        case sam2Tiny = "SAM2 Tiny (HF)"  // 新規追加
        
        var displayName: String { rawValue }
    }
    
    private func initializeSession() {
        switch selectedModelType {
        case .sam2Tiny:
            do {
                sam2Session = try Sam2Session(modelName: "SAM2Tiny")
                session = nil
            } catch {
                print("Failed to initialize SAM2: \(error)")
            }
        case .mobileSAM, .sam2_1_base:
            do {
                let modelRef = selectedModelType == .mobileSAM ? 
                    SamModelRef.bundled(.mobileSAM) : 
                    SamModelRef.bundled(.sam2_1_base)
                session = try SamSession(model: modelRef)
                sam2Session = nil
            } catch {
                print("Failed to initialize session: \(error)")
            }
        }
    }
    
    private func runSegmentation() {
        guard let image = selectedImage else { return }
        
        Task {
            do {
                if let sam2Session = sam2Session {
                    // SAM2での推論
                    try sam2Session.setImage(image.cgImage!)
                    let result = try await sam2Session.predict(points: points)
                    await updateResult(result)
                } else if let session = session {
                    // 従来モデルでの推論
                    try session.setImage(image.cgImage!)
                    let result = try await session.predict(points: points)
                    await updateResult(result)
                }
            } catch {
                print("Segmentation failed: \(error)")
            }
        }
    }
}
```

## 4. パフォーマンス比較

| モデル | サイズ | 推論速度 | メモリ使用量 | 精度 |
|--------|--------|----------|-------------|------|
| MobileSAM | ~40MB | 高速 | 低 | 中 |
| SAM2 Tiny (HF) | ~50MB | 高速 | 低 | 高 |
| SAM 2.1 Base | ~320MB | 中速 | 高 | 最高 |

## 5. トラブルシューティング

### よくある問題

1. **モデルが見つからない**
   ```
   Error: modelNotFound
   ```
   - 3つの`.mlpackage`ファイルがプロジェクトに正しく追加されているか確認
   - ファイル名が正確か確認（大文字小文字含む）

2. **メモリ不足**
   ```
   Error: Memory allocation failed
   ```
   - より小さい画像サイズを使用
   - バックグラウンドアプリを終了

3. **推論が遅い**
   - `RuntimeConfig.bestAvailable`を使用してGPU加速を有効化
   - 画像解像度を下げる

### デバッグ用コード

```swift
// モデルファイルの存在確認
private func checkModelFiles() {
    let bundle = Bundle.main
    let models = ["SAM2TinyImageEncoderFLOAT16", 
                  "SAM2TinyPromptEncoderFLOAT16", 
                  "SAM2TinyMaskDecoderFLOAT16"]
    
    for model in models {
        if bundle.url(forResource: model, withExtension: "mlpackage") != nil {
            print("✓ \(model) found")
        } else {
            print("✗ \(model) missing")
        }
    }
}
```

## 6. 次のステップ

- [ ] より大きなSAM2モデル（Small/Base/Large）の対応
- [ ] Android版での同等機能の実装
- [ ] ビデオセグメンテーション対応
- [ ] リアルタイム処理の最適化

HuggingFaceのSAM2 Tinyモデルにより、軽量でありながら高精度なセグメンテーションが可能になります。