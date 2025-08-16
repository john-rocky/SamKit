import SwiftUI
import PhotosUI
import SAMKit
import SAMKitUI

struct ContentView: View {
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var selectedModel: ModelType = .mobileSam
    @State private var selectedSam2Model: Sam2ModelType = .tiny
    @State private var useSeparateSam2 = false
    @State private var showSegmentationView = false
    @State private var isLoadingModel = false
    @State private var errorMessage: String?
    @State private var showError = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "camera.metering.multispot")
                        .font(.system(size: 60))
                        .foregroundColor(.blue)
                    
                    Text("SAMKit Demo")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Segment Anything on Mobile")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 40)
                
                // Selected Image Preview
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .cornerRadius(12)
                        .shadow(radius: 5)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue.opacity(0.5), lineWidth: 2)
                        )
                        .padding(.horizontal)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.gray.opacity(0.1))
                        .frame(height: 300)
                        .overlay(
                            VStack(spacing: 12) {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.system(size: 50))
                                    .foregroundColor(.gray)
                                Text("Select an image to segment")
                                    .foregroundColor(.secondary)
                            }
                        )
                        .padding(.horizontal)
                }
                
                // Model Selection
                VStack(alignment: .leading, spacing: 12) {
                    Text("Model")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    VStack(spacing: 8) {
                        Toggle("Use HuggingFace SAM2 Models", isOn: $useSeparateSam2)
                            .toggleStyle(SwitchToggleStyle())
                        
                        if useSeparateSam2 {
                            Picker("SAM2 Model", selection: $selectedSam2Model) {
                                Text("SAM2 Tiny (HF)").tag(Sam2ModelType.tiny)
                                Text("SAM2 Small (HF)").tag(Sam2ModelType.small)
                                Text("SAM2 Base (HF)").tag(Sam2ModelType.base)
                                Text("SAM2 Large (HF)").tag(Sam2ModelType.large)
                            }
                            .pickerStyle(MenuPickerStyle())
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                        } else {
                            Picker("Model", selection: $selectedModel) {
                                Text("MobileSAM (Fast)").tag(ModelType.mobileSam)
                                Text("SAM 2.1 Base").tag(ModelType.sam2_1_base)
                                Text("SAM 2.1 Large").tag(ModelType.sam2_1_large)
                            }
                            .pickerStyle(SegmentedPickerStyle())
                        }
                    }
                }
                .padding(.horizontal)
                
                Spacer()
                
                // Action Buttons
                VStack(spacing: 12) {
                    // Image Selection Buttons
                    HStack(spacing: 12) {
                        Button(action: {
                            showImagePicker = true
                        }) {
                            Label("Photo Library", systemImage: "photo.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button(action: {
                            showCamera = true
                        }) {
                            Label("Camera", systemImage: "camera.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }
                    
                    // Segment Button
                    Button(action: {
                        startSegmentation()
                    }) {
                        if isLoadingModel {
                            HStack {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                                Text("Loading Model...")
                            }
                            .frame(maxWidth: .infinity)
                        } else {
                            Label("Start Segmentation", systemImage: "wand.and.stars")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(selectedImage == nil || isLoadingModel)
                }
                .padding(.horizontal)
                .padding(.bottom, 20)
            }
            .navigationBarHidden(true)
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(image: $selectedImage, sourceType: .photoLibrary)
            }
            .sheet(isPresented: $showCamera) {
                ImagePicker(image: $selectedImage, sourceType: .camera)
            }
            .fullScreenCover(isPresented: $showSegmentationView) {
                if let image = selectedImage {
                    if useSeparateSam2 {
                        Sam2SegmentationView(
                            image: image,
                            sam2ModelType: selectedSam2Model
                        )
                    } else {
                        SegmentationView(
                            image: image,
                            modelType: selectedModel
                        )
                    }
                }
            }
            .alert("Error", isPresented: $showError) {
                Button("OK") {
                    errorMessage = nil
                }
            } message: {
                Text(errorMessage ?? "An error occurred")
            }
        }
    }
    
    private func startSegmentation() {
        guard selectedImage != nil else { return }
        
        isLoadingModel = true
        
        // Load model and start segmentation
        Task {
            do {
                if useSeparateSam2 {
                    // Check if SAM2 models are available
                    _ = try await loadSam2Model(selectedSam2Model)
                } else {
                    // Check if traditional model is available
                    _ = try await loadModel(selectedModel)
                }
                
                await MainActor.run {
                    isLoadingModel = false
                    showSegmentationView = true
                }
            } catch {
                await MainActor.run {
                    isLoadingModel = false
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
    
    private func loadModel(_ type: ModelType) async throws -> SamModelRef {
        // In a real app, you might download the model here
        // For demo, we assume models are bundled
        return try SamModelRef.bundled(type)
    }
    
    private func loadSam2Model(_ type: Sam2ModelType) async throws -> Sam2ModelRef {
        // Check if HuggingFace SAM2 models are available
        return try Sam2ModelRef.fromHuggingFace(modelType: type)
    }
}

// MARK: - Segmentation View

struct SegmentationView: View {
    let image: UIImage
    let modelType: ModelType
    @Environment(\.dismiss) private var dismiss
    @State private var modelRef: SamModelRef?
    @State private var isLoading = true
    @State private var loadError: Error?
    
    var body: some View {
        NavigationView {
            ZStack {
                if let modelRef = modelRef {
                    SamView(
                        image: image,
                        model: modelRef,
                        config: .bestAvailable
                    )
                } else if isLoading {
                    VStack {
                        ProgressView()
                        Text("Loading \(modelType.displayName)...")
                            .padding(.top)
                    }
                } else if let error = loadError {
                    VStack {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 50))
                            .foregroundColor(.red)
                        Text("Failed to load model")
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
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .principal) {
                    Text(modelType.displayName)
                        .font(.headline)
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: {
                            // Share functionality
                        }) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        
                        Button(action: {
                            // Save functionality
                        }) {
                            Label("Save", systemImage: "square.and.arrow.down")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
        }
        .onAppear {
            loadModel()
        }
    }
    
    private func loadModel() {
        Task {
            do {
                let ref = try SamModelRef.bundled(modelType)
                await MainActor.run {
                    self.modelRef = ref
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.loadError = error
                    self.isLoading = false
                }
            }
        }
    }
}

// MARK: - Image Picker

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    let sourceType: UIImagePickerController.SourceType
    @Environment(\.presentationMode) private var presentationMode
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        picker.allowsEditing = false
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController,
                                 didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                // Fix orientation for photos taken with camera
                parent.image = image.fixedOrientation()
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - Model Type Extension

extension ModelType {
    var displayName: String {
        switch self {
        case .mobileSam:
            return "MobileSAM"
        case .sam2_1_tiny:
            return "SAM 2.1 Tiny"
        case .sam2_1_small:
            return "SAM 2.1 Small"
        case .sam2_1_base:
            return "SAM 2.1 Base"
        case .sam2_1_large:
            return "SAM 2.1 Large"
        case .sam2_1_basePlus:
            return "SAM 2.1 Base+"
        }
    }
}

// MARK: - SAM2 Segmentation View

struct Sam2SegmentationView: View {
    let image: UIImage
    let sam2ModelType: Sam2ModelType
    @Environment(\.dismiss) private var dismiss
    @State private var sam2Session: Sam2Session?
    @State private var isLoading = true
    @State private var loadError: Error?
    
    var body: some View {
        NavigationView {
            ZStack {
                if let sam2Session = sam2Session {
                    Sam2InteractiveView(
                        image: image,
                        session: sam2Session
                    )
                } else if isLoading {
                    VStack {
                        ProgressView()
                        Text("Loading \(sam2ModelType.displayName)...")
                            .padding(.top)
                        Text("Initializing separate encoder models")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } else if let error = loadError {
                    VStack {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 50))
                            .foregroundColor(.red)
                        Text("Failed to load SAM2 models")
                            .font(.headline)
                            .padding(.top)
                        Text(error.localizedDescription)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding()
                        
                        Text("Make sure all three .mlpackage files are added to the project")
                            .font(.caption)
                            .foregroundColor(.orange)
                            .multilineTextAlignment(.center)
                            .padding()
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .principal) {
                    Text(sam2ModelType.displayName + " (HF)")
                        .font(.headline)
                }
            }
        }
        .onAppear {
            loadSam2Model()
        }
    }
    
    private func loadSam2Model() {
        Task {
            do {
                // Use Neural Engine optimized configuration
                let optimizedConfig = RuntimeConfig(
                    computeUnits: .neuralEnginePreferred,
                    enableFP16: true
                )
                
                let session = try Sam2Session(
                    modelName: sam2ModelType.modelPrefix,
                    config: optimizedConfig
                )
                
                await MainActor.run {
                    self.sam2Session = session
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.loadError = error
                    self.isLoading = false
                }
            }
        }
    }
}

// MARK: - SAM2 Interactive View

// MARK: - Input Mode Enum

enum InputMode: String, CaseIterable {
    case point = "Point"
    case box = "Box"
    case both = "Both"
    
    var icon: String {
        switch self {
        case .point: return "hand.point.up.left"
        case .box: return "rectangle.dashed"
        case .both: return "rectangle.and.hand.point.up.left"
        }
    }
}

struct Sam2InteractiveView: View {
    let image: UIImage
    let session: Sam2Session
    
    @State private var points: [SamPoint] = []
    @State private var boundingBox: SamBox?
    @State private var result: SamResult?
    @State private var isProcessing = false
    @State private var showMask = true
    @State private var selectedMaskIndex = 0
    @State private var imageSize: CGSize = .zero
    @State private var inputMode: InputMode = .point
    @State private var dragStart: CGPoint?
    @State private var dragEnd: CGPoint?
    @State private var showNegativePoints = false
    
    var body: some View {
        VStack {
            GeometryReader { geometry in
                ZStack {
                    // Base image
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .background(
                            GeometryReader { imgGeometry in
                                Color.clear
                                    .onAppear {
                                        imageSize = imgGeometry.size
                                    }
                            }
                        )
                        .contentShape(Rectangle())
                        .gesture(
                            inputMode == .box || inputMode == .both ?
                            DragGesture()
                                .onChanged { value in
                                    if dragStart == nil {
                                        dragStart = value.startLocation
                                    }
                                    dragEnd = value.location
                                }
                                .onEnded { value in
                                    if let start = dragStart {
                                        // Convert to image coordinates
                                        let startPoint = convertToImageCoordinates(
                                            tapLocation: start,
                                            viewSize: geometry.size,
                                            imageSize: image.size
                                        )
                                        let endPoint = convertToImageCoordinates(
                                            tapLocation: value.location,
                                            viewSize: geometry.size,
                                            imageSize: image.size
                                        )
                                        
                                        // Create bounding box
                                        let minX = min(startPoint.x, endPoint.x)
                                        let minY = min(startPoint.y, endPoint.y)
                                        let maxX = max(startPoint.x, endPoint.x)
                                        let maxY = max(startPoint.y, endPoint.y)
                                        
                                        boundingBox = SamBox(
                                            x0: Float(minX),
                                            y0: Float(minY),
                                            x1: Float(maxX),
                                            y1: Float(maxY)
                                        )
                                        
                                        dragStart = nil
                                        dragEnd = nil
                                        runSegmentation()
                                    }
                                }
                            : nil
                        )
                        .onTapGesture { location in
                            if inputMode == .point || inputMode == .both {
                                // Convert tap location to image coordinates
                                let imagePoint = convertToImageCoordinates(
                                    tapLocation: location,
                                    viewSize: geometry.size,
                                    imageSize: image.size
                                )
                                addPoint(at: imagePoint, isPositive: !showNegativePoints)
                            }
                        }
                    
                    // Overlay mask - only show the selected mask
                    if showMask, let result = result {
                        // Ensure selectedMaskIndex is valid
                        let safeIndex = min(selectedMaskIndex, result.masks.count - 1)
                        if safeIndex >= 0 && safeIndex < result.masks.count {
                            Image(uiImage: UIImage(cgImage: result.masks[safeIndex].cgImage))
                                .resizable()
                                .scaledToFit()
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .opacity(0.6)
                                .allowsHitTesting(false)
                        }
                    }
                    
                    // Box overlay during drag
                    if let start = dragStart, let end = dragEnd {
                        Rectangle()
                            .stroke(Color.blue, lineWidth: 2)
                            .background(Color.blue.opacity(0.1))
                            .frame(
                                width: abs(end.x - start.x),
                                height: abs(end.y - start.y)
                            )
                            .position(
                                x: min(start.x, end.x) + abs(end.x - start.x) / 2,
                                y: min(start.y, end.y) + abs(end.y - start.y) / 2
                            )
                    }
                    
                    // Bounding box overlay
                    if let box = boundingBox {
                        let topLeft = convertFromImageCoordinates(
                            imagePoint: CGPoint(x: CGFloat(box.x0), y: CGFloat(box.y0)),
                            viewSize: geometry.size,
                            imageSize: image.size
                        )
                        let bottomRight = convertFromImageCoordinates(
                            imagePoint: CGPoint(x: CGFloat(box.x1), y: CGFloat(box.y1)),
                            viewSize: geometry.size,
                            imageSize: image.size
                        )
                        
                        Rectangle()
                            .stroke(Color.blue, lineWidth: 2)
                            .background(Color.clear)
                            .frame(
                                width: abs(bottomRight.x - topLeft.x),
                                height: abs(bottomRight.y - topLeft.y)
                            )
                            .position(
                                x: topLeft.x + (bottomRight.x - topLeft.x) / 2,
                                y: topLeft.y + (bottomRight.y - topLeft.y) / 2
                            )
                    }
                    
                    // Points overlay
                    ForEach(Array(points.enumerated()), id: \.offset) { index, point in
                        Circle()
                            .fill(point.label == .positive ? Color.green : Color.red)
                            .frame(width: 12, height: 12)
                            .overlay(
                                Circle()
                                    .stroke(Color.white, lineWidth: 2)
                            )
                            .position(
                                convertFromImageCoordinates(
                                    imagePoint: CGPoint(x: point.x, y: point.y),
                                    viewSize: geometry.size,
                                    imageSize: image.size
                                )
                            )
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
                // Input mode selector
                Picker("Input Mode", selection: $inputMode) {
                    ForEach(InputMode.allCases, id: \.self) { mode in
                        Label(mode.rawValue, systemImage: mode.icon)
                            .tag(mode)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                
                // Action buttons
                HStack {
                    Button(action: {
                        points.removeAll()
                        boundingBox = nil
                        result = nil
                    }) {
                        Label("Clear All", systemImage: "trash")
                            .font(.caption)
                    }
                    .disabled(points.isEmpty && boundingBox == nil)
                    
                    if inputMode == .point || inputMode == .both {
                        Divider()
                            .frame(height: 20)
                        
                        Toggle(isOn: $showNegativePoints) {
                            Label(
                                showNegativePoints ? "Negative" : "Positive",
                                systemImage: showNegativePoints ? "minus.circle" : "plus.circle"
                            )
                            .font(.caption)
                        }
                        .toggleStyle(.button)
                    }
                    
                    Spacer()
                    
                    Toggle("Show Mask", isOn: $showMask)
                        .disabled(result == nil)
                }
                
                if let result = result, result.masks.count > 1 {
                    VStack {
                        Text("Masks (\(result.masks.count))")
                            .font(.caption)
                        
                        // Limit picker to max 3 segments for better UI
                        let maskCount = min(result.masks.count, 3)
                        Picker("Mask", selection: $selectedMaskIndex) {
                            ForEach(0..<maskCount, id: \.self) { index in
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
    
    private func addPoint(at location: CGPoint, isPositive: Bool = true) {
        let point = SamPoint(
            x: location.x,
            y: location.y,
            label: isPositive ? .positive : .negative
        )
        points.append(point)
        runSegmentation()
    }
    
    private func runSegmentation() {
        guard !points.isEmpty || boundingBox != nil else { return }
        
        isProcessing = true
        
        Task {
            do {
                if result == nil {
                    // Set image for first time
                    try session.setImage(image.cgImage!)
                }
                
                let newResult = try await session.predict(
                    points: points,
                    box: boundingBox
                )
                
                await MainActor.run {
                    self.result = newResult
                    self.selectedMaskIndex = 0
                    self.isProcessing = false
                    
                    // Debug: print mask info
                    print("Got \(newResult.masks.count) masks")
                    for (i, mask) in newResult.masks.enumerated() {
                        print("Mask \(i): \(mask.width)x\(mask.height), score: \(mask.score)")
                    }
                }
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                }
                print("Segmentation failed: \(error)")
            }
        }
    }
    
    private func convertToImageCoordinates(tapLocation: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        // Calculate the actual displayed image size (maintaining aspect ratio)
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height
        
        var displayedSize: CGSize
        var offset: CGPoint
        
        if imageAspect > viewAspect {
            // Image is wider, fit to width
            displayedSize = CGSize(width: viewSize.width, height: viewSize.width / imageAspect)
            offset = CGPoint(x: 0, y: (viewSize.height - displayedSize.height) / 2)
        } else {
            // Image is taller, fit to height
            displayedSize = CGSize(width: viewSize.height * imageAspect, height: viewSize.height)
            offset = CGPoint(x: (viewSize.width - displayedSize.width) / 2, y: 0)
        }
        
        // Convert tap location to image coordinates
        let relativeX = (tapLocation.x - offset.x) / displayedSize.width
        let relativeY = (tapLocation.y - offset.y) / displayedSize.height
        
        // Clamp to valid range
        let clampedX = max(0, min(1, relativeX))
        let clampedY = max(0, min(1, relativeY))
        
        return CGPoint(
            x: clampedX * imageSize.width,
            y: clampedY * imageSize.height
        )
    }
    
    private func convertFromImageCoordinates(imagePoint: CGPoint, viewSize: CGSize, imageSize: CGSize) -> CGPoint {
        // Calculate the actual displayed image size (maintaining aspect ratio)
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height
        
        var displayedSize: CGSize
        var offset: CGPoint
        
        if imageAspect > viewAspect {
            // Image is wider, fit to width
            displayedSize = CGSize(width: viewSize.width, height: viewSize.width / imageAspect)
            offset = CGPoint(x: 0, y: (viewSize.height - displayedSize.height) / 2)
        } else {
            // Image is taller, fit to height
            displayedSize = CGSize(width: viewSize.height * imageAspect, height: viewSize.height)
            offset = CGPoint(x: (viewSize.width - displayedSize.width) / 2, y: 0)
        }
        
        // Convert image coordinates to view coordinates
        let viewX = (imagePoint.x / imageSize.width) * displayedSize.width + offset.x
        let viewY = (imagePoint.y / imageSize.height) * displayedSize.height + offset.y
        
        return CGPoint(x: viewX, y: viewY)
    }
}

// MARK: - UIImage Extension for Orientation Fix

extension UIImage {
    func fixedOrientation() -> UIImage {
        // If image orientation is already correct, return as is
        if imageOrientation == .up {
            return self
        }
        
        // Calculate the proper transformation
        var transform = CGAffineTransform.identity
        
        switch imageOrientation {
        case .down, .downMirrored:
            transform = transform.translatedBy(x: size.width, y: size.height)
            transform = transform.rotated(by: .pi)
            
        case .left, .leftMirrored:
            transform = transform.translatedBy(x: size.width, y: 0)
            transform = transform.rotated(by: .pi / 2)
            
        case .right, .rightMirrored:
            transform = transform.translatedBy(x: 0, y: size.height)
            transform = transform.rotated(by: -.pi / 2)
            
        case .up, .upMirrored:
            break
            
        @unknown default:
            break
        }
        
        switch imageOrientation {
        case .upMirrored, .downMirrored:
            transform = transform.translatedBy(x: size.width, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
            
        case .leftMirrored, .rightMirrored:
            transform = transform.translatedBy(x: size.height, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
            
        case .up, .down, .left, .right:
            break
            
        @unknown default:
            break
        }
        
        // Create a new context with the correct size
        let ctx = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: cgImage!.bitsPerComponent,
            bytesPerRow: 0,
            space: cgImage!.colorSpace!,
            bitmapInfo: cgImage!.bitmapInfo.rawValue
        )!
        
        ctx.concatenate(transform)
        
        switch imageOrientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            ctx.draw(cgImage!, in: CGRect(x: 0, y: 0, width: size.height, height: size.width))
        default:
            ctx.draw(cgImage!, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        }
        
        // Create a new UIImage from the drawing context
        let cgImg = ctx.makeImage()!
        return UIImage(cgImage: cgImg)
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}