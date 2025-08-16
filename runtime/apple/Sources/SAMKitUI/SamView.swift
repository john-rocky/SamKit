import SwiftUI
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
    
    var body: some View {
        VStack {
            ZStack {
                // Base image
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .onTapGesture { 
                        // Use a simple center point for tap gesture
                        addPoint(at: CGPoint(x: 400, y: 400))
                    }
                
                // Overlay mask
                if showMask, let result = result, result.masks.indices.contains(selectedMaskIndex) {
                    Image(uiImage: UIImage(cgImage: result.masks[selectedMaskIndex].cgImage))
                        .resizable()
                        .scaledToFit()
                        .opacity(0.6)
                }
                
                // Points overlay
                ForEach(Array(points.enumerated()), id: \.offset) { index, point in
                    Circle()
                        .fill(point.label == .positive ? Color.green : Color.red)
                        .frame(width: 12, height: 12)
                        .position(x: point.x, y: point.y)
                }
                
                if isProcessing {
                    Color.black.opacity(0.3)
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(1.5)
                }
            }
            
            // Controls
            VStack(spacing: 16) {
                HStack {
                    Button("Clear Points") {
                        points.removeAll()
                        result = nil
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
    
    private func addPoint(at location: CGPoint) {
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
                
                let newResult = try await session.predict(points: points)
                
                await MainActor.run {
                    self.result = newResult
                    self.selectedMaskIndex = 0
                    self.isProcessing = false
                }
            } catch {
                await MainActor.run {
                    self.isProcessing = false
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