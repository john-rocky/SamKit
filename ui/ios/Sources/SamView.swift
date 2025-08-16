import SwiftUI
import SAMKit
import Combine

/// Interactive view for SAM segmentation on iOS
public struct SamView: View {
    @StateObject private var viewModel: SamViewModel
    @State private var currentImage: UIImage?
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero
    
    private let debounceDelay: Double = 0.08 // 80ms debounce
    
    public init(
        image: UIImage,
        model: SamModelRef,
        config: RuntimeConfig = .bestAvailable
    ) {
        self.currentImage = image
        self._viewModel = StateObject(wrappedValue: SamViewModel(
            image: image,
            model: model,
            config: config
        ))
    }
    
    public var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color.black.edgesIgnoringSafeArea(.all)
                
                // Image and mask overlay
                if let image = currentImage {
                    ImageCanvas(
                        image: image,
                        masks: viewModel.masks,
                        points: viewModel.points,
                        box: viewModel.currentBox,
                        selectedMaskIndex: viewModel.selectedMaskIndex,
                        scale: scale,
                        offset: offset,
                        onTap: handleTap,
                        onDrag: handleDrag
                    )
                }
                
                // Controls overlay
                VStack {
                    Spacer()
                    ControlPanel(viewModel: viewModel)
                        .padding()
                }
            }
            .gesture(magnificationGesture)
            .gesture(dragGesture)
        }
        .onAppear {
            viewModel.setImage(currentImage!)
        }
    }
    
    // MARK: - Gestures
    
    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                scale = lastScale * value
            }
            .onEnded { _ in
                lastScale = scale
            }
    }
    
    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                offset = CGSize(
                    width: lastOffset.width + value.translation.width,
                    height: lastOffset.height + value.translation.height
                )
            }
            .onEnded { _ in
                lastOffset = offset
            }
    }
    
    private func handleTap(location: CGPoint) {
        let imagePoint = convertToImageCoordinates(location)
        viewModel.addPoint(at: imagePoint)
    }
    
    private func handleDrag(start: CGPoint, end: CGPoint) {
        let startImage = convertToImageCoordinates(start)
        let endImage = convertToImageCoordinates(end)
        
        viewModel.setBox(
            from: startImage,
            to: endImage
        )
    }
    
    private func convertToImageCoordinates(_ point: CGPoint) -> CGPoint {
        // Convert from view coordinates to image coordinates
        // accounting for scale and offset
        guard let image = currentImage else { return point }
        
        let imageSize = image.size
        let scaledSize = CGSize(
            width: imageSize.width * scale,
            height: imageSize.height * scale
        )
        
        let adjustedPoint = CGPoint(
            x: (point.x - offset.width) / scale,
            y: (point.y - offset.height) / scale
        )
        
        return adjustedPoint
    }
}

// MARK: - Image Canvas

struct ImageCanvas: View {
    let image: UIImage
    let masks: [SamMask]
    let points: [SamPoint]
    let box: SamBox?
    let selectedMaskIndex: Int
    let scale: CGFloat
    let offset: CGSize
    let onTap: (CGPoint) -> Void
    let onDrag: (CGPoint, CGPoint) -> Void
    
    @State private var dragStart: CGPoint?
    
    var body: some View {
        Canvas { context, size in
            // Draw image
            if let cgImage = image.cgImage {
                context.draw(
                    Image(cgImage, scale: 1.0, label: Text("Image")),
                    in: CGRect(origin: .zero, size: size)
                )
            }
            
            // Draw masks
            for (index, mask) in masks.enumerated() {
                if index == selectedMaskIndex || selectedMaskIndex == -1 {
                    drawMask(mask, in: context, size: size, opacity: 0.5)
                }
            }
            
            // Draw points
            for point in points {
                drawPoint(point, in: context, size: size)
            }
            
            // Draw box
            if let box = box {
                drawBox(box, in: context, size: size)
            }
        }
        .scaleEffect(scale)
        .offset(offset)
        .onTapGesture { location in
            onTap(location)
        }
        .gesture(
            DragGesture()
                .onChanged { value in
                    if dragStart == nil {
                        dragStart = value.startLocation
                    }
                }
                .onEnded { value in
                    if let start = dragStart {
                        onDrag(start, value.location)
                        dragStart = nil
                    }
                }
        )
    }
    
    private func drawMask(_ mask: SamMask, in context: GraphicsContext, size: CGSize, opacity: Double) {
        guard let maskImage = mask.toColoredCGImage(
            color: CGColor(red: 0, green: 0.5, blue: 1, alpha: 1)
        ) else { return }
        
        context.opacity = opacity
        context.draw(
            Image(maskImage, scale: 1.0, label: Text("Mask")),
            in: CGRect(origin: .zero, size: size)
        )
        context.opacity = 1.0
    }
    
    private func drawPoint(_ point: SamPoint, in context: GraphicsContext, size: CGSize) {
        let color = point.label == 1 ? Color.green : Color.red
        let position = CGPoint(x: CGFloat(point.x), y: CGFloat(point.y))
        
        context.fill(
            Circle().path(in: CGRect(
                x: position.x - 5,
                y: position.y - 5,
                width: 10,
                height: 10
            )),
            with: .color(color)
        )
    }
    
    private func drawBox(_ box: SamBox, in context: GraphicsContext, size: CGSize) {
        let rect = CGRect(
            x: CGFloat(box.x0),
            y: CGFloat(box.y0),
            width: CGFloat(box.x1 - box.x0),
            height: CGFloat(box.y1 - box.y0)
        )
        
        context.stroke(
            Rectangle().path(in: rect),
            with: .color(.yellow),
            lineWidth: 2
        )
    }
}

// MARK: - Control Panel

struct ControlPanel: View {
    @ObservedObject var viewModel: SamViewModel
    
    var body: some View {
        VStack(spacing: 12) {
            // Mask selection
            if viewModel.masks.count > 1 {
                Picker("Mask", selection: $viewModel.selectedMaskIndex) {
                    Text("All").tag(-1)
                    ForEach(0..<viewModel.masks.count, id: \.self) { index in
                        Text("Mask \(index + 1)").tag(index)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
            }
            
            // Threshold slider
            HStack {
                Text("Threshold")
                Slider(
                    value: $viewModel.maskThreshold,
                    in: -10...10,
                    onEditingChanged: { editing in
                        if !editing {
                            viewModel.updateThreshold()
                        }
                    }
                )
                Text(String(format: "%.1f", viewModel.maskThreshold))
            }
            
            // Action buttons
            HStack(spacing: 16) {
                Button("Clear Points") {
                    viewModel.clearPoints()
                }
                
                Button("Clear Box") {
                    viewModel.clearBox()
                }
                
                Button("Undo") {
                    viewModel.undo()
                }
                .disabled(!viewModel.canUndo)
                
                Button("Export") {
                    viewModel.exportMask()
                }
                .disabled(viewModel.masks.isEmpty)
            }
            .buttonStyle(.bordered)
        }
        .padding()
        .background(Color.black.opacity(0.8))
        .cornerRadius(12)
    }
}

// MARK: - View Model

@MainActor
class SamViewModel: ObservableObject {
    @Published var masks: [SamMask] = []
    @Published var points: [SamPoint] = []
    @Published var currentBox: SamBox?
    @Published var selectedMaskIndex: Int = -1
    @Published var maskThreshold: Float = 0.0
    @Published var isProcessing: Bool = false
    @Published var canUndo: Bool = false
    
    private var session: SamSession?
    private let model: SamModelRef
    private let config: RuntimeConfig
    private var currentImage: UIImage?
    private var debounceTimer: Timer?
    private var history: [(points: [SamPoint], box: SamBox?)] = []
    
    init(image: UIImage, model: SamModelRef, config: RuntimeConfig) {
        self.currentImage = image
        self.model = model
        self.config = config
        setupSession()
    }
    
    private func setupSession() {
        do {
            session = try SamSession(model: model, config: config)
        } catch {
            print("Failed to create session: \(error)")
        }
    }
    
    func setImage(_ image: UIImage) {
        currentImage = image
        clearAll()
        
        guard let cgImage = image.cgImage else { return }
        
        Task {
            isProcessing = true
            do {
                try session?.setImage(cgImage)
            } catch {
                print("Failed to set image: \(error)")
            }
            isProcessing = false
        }
    }
    
    func addPoint(at location: CGPoint) {
        let point = SamPoint(
            x: Float(location.x),
            y: Float(location.y),
            label: 1 // Default to positive
        )
        points.append(point)
        saveHistory()
        debouncedPredict()
    }
    
    func setBox(from start: CGPoint, to end: CGPoint) {
        currentBox = SamBox(
            x0: Float(min(start.x, end.x)),
            y0: Float(min(start.y, end.y)),
            x1: Float(max(start.x, end.x)),
            y1: Float(max(start.y, end.y))
        )
        saveHistory()
        debouncedPredict()
    }
    
    func clearPoints() {
        points.removeAll()
        debouncedPredict()
    }
    
    func clearBox() {
        currentBox = nil
        debouncedPredict()
    }
    
    func clearAll() {
        points.removeAll()
        currentBox = nil
        masks.removeAll()
        history.removeAll()
        canUndo = false
    }
    
    func undo() {
        guard history.count > 1 else { return }
        history.removeLast()
        if let last = history.last {
            points = last.points
            currentBox = last.box
            debouncedPredict()
        }
        canUndo = history.count > 1
    }
    
    func updateThreshold() {
        // Re-apply threshold to existing masks if we have logits
        // This would require keeping the original logits
        debouncedPredict()
    }
    
    func exportMask() {
        guard let mask = selectedMask else { return }
        // Export logic here - save to photos, share, etc.
        if let cgImage = mask.toCGImage() {
            UIImageWriteToSavedPhotosAlbum(
                UIImage(cgImage: cgImage),
                nil,
                nil,
                nil
            )
        }
    }
    
    private var selectedMask: SamMask? {
        if selectedMaskIndex == -1 && !masks.isEmpty {
            return masks.first
        } else if selectedMaskIndex >= 0 && selectedMaskIndex < masks.count {
            return masks[selectedMaskIndex]
        }
        return nil
    }
    
    private func saveHistory() {
        history.append((points: points, box: currentBox))
        canUndo = history.count > 1
    }
    
    private func debouncedPredict() {
        debounceTimer?.invalidate()
        debounceTimer = Timer.scheduledTimer(withTimeInterval: 0.08, repeats: false) { _ in
            Task { @MainActor in
                await self.predict()
            }
        }
    }
    
    private func predict() async {
        guard !points.isEmpty || currentBox != nil else {
            masks.removeAll()
            return
        }
        
        isProcessing = true
        
        do {
            let result = try await session?.predict(
                points: points,
                box: currentBox,
                maskInput: nil,
                options: SamOptions(
                    multimaskOutput: true,
                    returnLogits: true,
                    maskThreshold: maskThreshold
                )
            )
            
            if let result = result {
                masks = result.masks
            }
        } catch {
            print("Prediction failed: \(error)")
        }
        
        isProcessing = false
    }
}