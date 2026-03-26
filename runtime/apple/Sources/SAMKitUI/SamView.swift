import SwiftUI
import UIKit
import SAMKit

public struct SamView: View {
    let image: UIImage
    let model: SamModelRef
    let config: RuntimeConfig
    
    @State private var session: SamSession?
    @State private var points: [SamPoint] = []
    @State private var result: SamResult?
    @State private var isProcessing = false
    @State private var showMask = true
    @State private var selectedMaskIndex = 0
    @State private var loadError: Error?
    
    public init(image: UIImage, model: SamModelRef, config: RuntimeConfig = .bestAvailable) {
        self.image = image
        self.model = model
        self.config = config
    }
    
    public var body: some View {
        VStack {
            if let error = loadError {
                ErrorView(error: error)
            } else if session != nil {
                InteractiveSegmentationView(
                    image: image,
                    session: session!,
                    points: $points,
                    result: $result,
                    isProcessing: $isProcessing,
                    showMask: $showMask,
                    selectedMaskIndex: $selectedMaskIndex
                )
            } else {
                LoadingView(modelType: model.modelType)
            }
        }
        .onAppear {
            loadSession()
        }
    }
    
    private func loadSession() {
        Task {
            do {
                let newSession = try SamSession(model: model, config: config)
                await MainActor.run {
                    self.session = newSession
                }
            } catch {
                await MainActor.run {
                    self.loadError = error
                }
            }
        }
    }
}

struct InteractiveSegmentationView: View {
    let image: UIImage
    let session: SamSession
    @Binding var points: [SamPoint]
    @Binding var result: SamResult?
    @Binding var isProcessing: Bool
    @Binding var showMask: Bool
    @Binding var selectedMaskIndex: Int
    @State private var errorMessage: String?
    
    var body: some View {
        VStack {
            GeometryReader { geometry in
                ZStack {
                    // Base image
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onEnded { value in
                                    let imagePoint = viewToImageCoordinates(
                                        value.location, viewSize: geometry.size
                                    )
                                    addPoint(at: imagePoint)
                                }
                        )

                    // Overlay mask
                    if showMask, let result = result, result.masks.indices.contains(selectedMaskIndex) {
                        Image(uiImage: UIImage(cgImage: result.masks[selectedMaskIndex].cgImage))
                            .resizable()
                            .scaledToFit()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .opacity(0.6)
                            .allowsHitTesting(false)
                    }

                    // Points overlay
                    ForEach(Array(points.enumerated()), id: \.offset) { index, point in
                        let viewPos = imageToViewCoordinates(
                            CGPoint(x: point.x, y: point.y), viewSize: geometry.size
                        )
                        Circle()
                            .fill(point.label == .positive ? Color.green : Color.red)
                            .frame(width: 12, height: 12)
                            .position(viewPos)
                    }

                    if isProcessing {
                        Color.black.opacity(0.3)
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)
                    }
                }
            }

            // Controls
            VStack(spacing: 16) {
                if let errorMessage = errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.red)
                        .lineLimit(3)
                }

                HStack {
                    Button("Clear Points") {
                        points.removeAll()
                        result = nil
                        errorMessage = nil
                    }
                    .disabled(points.isEmpty)

                    Spacer()

                    Toggle("Show Mask", isOn: $showMask)
                        .disabled(result == nil)
                }

                if let result = result, result.masks.count > 1 {
                    VStack {
                        Text("Masks (\(result.masks.count))")
                            .font(.caption)

                        Picker("Mask", selection: $selectedMaskIndex) {
                            ForEach(0..<result.masks.count, id: \.self) { index in
                                Text("Mask \(index + 1) (\(String(format: "%.3f", result.scores[index])))")
                                    .tag(index)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                    }
                }
            }
            .padding()
        }
    }

    // MARK: - Coordinate Conversion

    /// Convert view tap coordinates to image coordinates (accounting for scaledToFit)
    private func viewToImageCoordinates(_ viewPoint: CGPoint, viewSize: CGSize) -> CGPoint {
        let imageSize = image.size
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height

        let displayedSize: CGSize
        let offset: CGPoint

        if imageAspect > viewAspect {
            // Image is wider: width fills, height is letterboxed
            let w = viewSize.width
            let h = w / imageAspect
            displayedSize = CGSize(width: w, height: h)
            offset = CGPoint(x: 0, y: (viewSize.height - h) / 2)
        } else {
            // Image is taller: height fills, width is pillarboxed
            let h = viewSize.height
            let w = h * imageAspect
            displayedSize = CGSize(width: w, height: h)
            offset = CGPoint(x: (viewSize.width - w) / 2, y: 0)
        }

        let x = (viewPoint.x - offset.x) / displayedSize.width * imageSize.width
        let y = (viewPoint.y - offset.y) / displayedSize.height * imageSize.height
        return CGPoint(
            x: min(max(x, 0), imageSize.width),
            y: min(max(y, 0), imageSize.height)
        )
    }

    /// Convert image coordinates back to view coordinates for display
    private func imageToViewCoordinates(_ imagePoint: CGPoint, viewSize: CGSize) -> CGPoint {
        let imageSize = image.size
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height

        let displayedSize: CGSize
        let offset: CGPoint

        if imageAspect > viewAspect {
            let w = viewSize.width
            let h = w / imageAspect
            displayedSize = CGSize(width: w, height: h)
            offset = CGPoint(x: 0, y: (viewSize.height - h) / 2)
        } else {
            let h = viewSize.height
            let w = h * imageAspect
            displayedSize = CGSize(width: w, height: h)
            offset = CGPoint(x: (viewSize.width - w) / 2, y: 0)
        }

        return CGPoint(
            x: imagePoint.x / imageSize.width * displayedSize.width + offset.x,
            y: imagePoint.y / imageSize.height * displayedSize.height + offset.y
        )
    }

    private func addPoint(at location: CGPoint) {
        if points.count >= PromptEncoder.maxPoints {
            // Drop the oldest point to stay within the model's limit
            points.removeFirst()
        }
        let point = SamPoint(x: location.x, y: location.y, label: .positive)
        points.append(point)
        runSegmentation()
    }
    
    private func runSegmentation() {
        guard !points.isEmpty else { return }
        
        isProcessing = true
        
        Task {
            do {
                if result == nil {
                    // Set image for first time
                    try session.setImage(image.cgImage!)
                }
                
                let newResult = try session.predict(points: points)
                
                await MainActor.run {
                    self.result = newResult
                    self.selectedMaskIndex = 0
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.errorMessage = error.localizedDescription
                }
                print("Segmentation failed: \(error)")
            }
        }
    }
}

struct LoadingView: View {
    let modelType: ModelType
    
    var body: some View {
        VStack {
            ProgressView()
            Text("Loading \(modelType.modelName)...")
                .padding(.top)
        }
    }
}

struct ErrorView: View {
    let error: Error
    
    var body: some View {
        VStack {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 50))
                .foregroundColor(.red)
            Text("Error")
                .font(.headline)
                .padding(.top)
            Text(error.localizedDescription)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding()
        }
    }
}