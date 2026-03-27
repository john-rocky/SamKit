import SwiftUI
import UIKit
import SAMKit
import SAMKitGrounding

// MARK: - Input Mode

enum InputMode: String, CaseIterable {
    case point = "Point"
    case box = "Box"
    case both = "Both"
    case text = "Text"

    var icon: String {
        switch self {
        case .point: return "hand.point.up.left"
        case .box: return "rectangle.dashed"
        case .both: return "rectangle.and.hand.point.up.left"
        case .text: return "text.magnifyingglass"
        }
    }
}

// MARK: - Unified Segmentation View

struct UnifiedSegmentationView: View {
    let image: UIImage
    @EnvironmentObject var modelManager: ModelManager
    @Environment(\.dismiss) private var dismiss

    // Interaction state
    @State private var points: [SamPoint] = []
    @State private var boundingBox: SamBox?
    @State private var samResult: SamResult?
    @State private var textResult: TextSegmentationResult?
    @State private var isProcessing = false
    @State private var showMask = true
    @State private var selectedMaskIndex = 0
    @State private var inputMode: InputMode = .point
    @State private var dragStart: CGPoint?
    @State private var dragEnd: CGPoint?
    @State private var showNegativePoints = false
    @State private var queryText: String = ""
    @State private var samImageSet = false
    @State private var sam2ImageSet = false
    @State private var textImageSet = false
    @State private var selectedTextIndices: Set<Int> = []
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Image + overlays
                GeometryReader { geometry in
                    ZStack {
                        // Base image
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(width: geometry.size.width, height: geometry.size.height)
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
                                        guard let start = dragStart else { return }
                                        let startPoint = viewToImage(start, viewSize: geometry.size)
                                        let endPoint = viewToImage(value.location, viewSize: geometry.size)

                                        let minX = min(startPoint.x, endPoint.x)
                                        let minY = min(startPoint.y, endPoint.y)
                                        let maxX = max(startPoint.x, endPoint.x)
                                        let maxY = max(startPoint.y, endPoint.y)

                                        boundingBox = SamBox(
                                            x0: Float(minX), y0: Float(minY),
                                            x1: Float(maxX), y1: Float(maxY)
                                        )

                                        dragStart = nil
                                        dragEnd = nil
                                        runSegmentation()
                                    }
                                : nil
                            )
                            .onTapGesture { location in
                                handleTap(at: location, geometry: geometry)
                            }

                        // SAM mask overlay (point/box modes)
                        if inputMode != .text, showMask, let result = samResult {
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

                        // Text mask overlays
                        if inputMode == .text, showMask, let result = textResult {
                            ForEach(Array(result.masks.enumerated()), id: \.offset) { index, mask in
                                let isSelected = selectedTextIndices.isEmpty || selectedTextIndices.contains(index)
                                if isSelected {
                                    Image(uiImage: UIImage(cgImage: mask.cgImage))
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: geometry.size.width, height: geometry.size.height)
                                        .opacity(0.5)
                                        .allowsHitTesting(false)
                                }
                            }
                        }

                        // Text detection bounding boxes
                        if inputMode == .text, let result = textResult {
                            ForEach(Array(result.detections.enumerated()), id: \.offset) { index, detection in
                                let isSelected = selectedTextIndices.isEmpty || selectedTextIndices.contains(index)
                                let topLeft = imageToView(
                                    CGPoint(x: CGFloat(detection.box.x0), y: CGFloat(detection.box.y0)),
                                    viewSize: geometry.size
                                )
                                let bottomRight = imageToView(
                                    CGPoint(x: CGFloat(detection.box.x1), y: CGFloat(detection.box.y1)),
                                    viewSize: geometry.size
                                )

                                Rectangle()
                                    .stroke(isSelected ? Color.green : Color.gray, lineWidth: isSelected ? 2 : 1)
                                    .frame(
                                        width: abs(bottomRight.x - topLeft.x),
                                        height: abs(bottomRight.y - topLeft.y)
                                    )
                                    .position(
                                        x: topLeft.x + (bottomRight.x - topLeft.x) / 2,
                                        y: topLeft.y + (bottomRight.y - topLeft.y) / 2
                                    )
                                    .onTapGesture { toggleTextSelection(index) }

                                Text("\(detection.label) \(String(format: "%.0f%%", detection.confidence * 100))")
                                    .font(.caption2)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 2)
                                    .background(isSelected ? Color.green : Color.gray)
                                    .cornerRadius(4)
                                    .position(x: topLeft.x + 40, y: topLeft.y - 8)
                            }
                        }

                        // Drag-in-progress box
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

                        // Committed bounding box
                        if inputMode != .text, let box = boundingBox {
                            let topLeft = imageToView(
                                CGPoint(x: CGFloat(box.x0), y: CGFloat(box.y0)),
                                viewSize: geometry.size
                            )
                            let bottomRight = imageToView(
                                CGPoint(x: CGFloat(box.x1), y: CGFloat(box.y1)),
                                viewSize: geometry.size
                            )
                            Rectangle()
                                .stroke(Color.blue, lineWidth: 2)
                                .frame(
                                    width: abs(bottomRight.x - topLeft.x),
                                    height: abs(bottomRight.y - topLeft.y)
                                )
                                .position(
                                    x: topLeft.x + (bottomRight.x - topLeft.x) / 2,
                                    y: topLeft.y + (bottomRight.y - topLeft.y) / 2
                                )
                        }

                        // Point markers
                        if inputMode != .text {
                            ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                                Circle()
                                    .fill(point.label == .positive ? Color.green : Color.red)
                                    .frame(width: 12, height: 12)
                                    .overlay(Circle().stroke(Color.white, lineWidth: 2))
                                    .position(
                                        imageToView(
                                            CGPoint(x: point.x, y: point.y),
                                            viewSize: geometry.size
                                        )
                                    )
                            }
                        }

                        // Processing indicator
                        if isProcessing {
                            Color.black.opacity(0.3)
                            VStack(spacing: 8) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(1.5)
                                if inputMode == .text {
                                    Text("Detecting...")
                                        .foregroundColor(.white)
                                        .font(.caption)
                                }
                            }
                        }

                        // Model not ready overlay
                        if !modelManager.isSamReady && inputMode != .text {
                            Color.black.opacity(0.5)
                            VStack {
                                ProgressView()
                                Text(modelManager.loadingStatus)
                                    .foregroundColor(.white)
                                    .font(.caption)
                            }
                        }
                    }
                }

                // Controls
                controlsPanel
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
                ToolbarItem(placement: .principal) {
                    Text(modelManager.activeModelName)
                        .font(.headline)
                }
            }
        }
        .task { await setImageOnSessions() }
        .onChange(of: inputMode) { _ in clearModeResults() }
    }

    // MARK: - Controls Panel

    @ViewBuilder
    private var controlsPanel: some View {
        VStack(spacing: 12) {
            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .lineLimit(3)
            }

            // Input mode picker
            Picker("Input Mode", selection: $inputMode) {
                ForEach(InputMode.allCases, id: \.self) { mode in
                    Label(mode.rawValue, systemImage: mode.icon).tag(mode)
                }
            }
            .pickerStyle(SegmentedPickerStyle())

            if inputMode == .text {
                textControls
            } else {
                pointBoxControls
            }
        }
        .padding()
    }

    @ViewBuilder
    private var pointBoxControls: some View {
        HStack {
            Button(action: clearAll) {
                Label("Clear All", systemImage: "trash")
                    .font(.caption)
            }
            .disabled(points.isEmpty && boundingBox == nil)

            if inputMode == .point || inputMode == .both {
                Divider().frame(height: 20)

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
                .disabled(samResult == nil)
        }

        if let result = samResult, result.masks.count > 1 {
            VStack {
                Text("Masks (\(result.masks.count))")
                    .font(.caption)
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

    @ViewBuilder
    private var textControls: some View {
        // Detection summary
        if let result = textResult, !result.detections.isEmpty {
            HStack {
                Text("\(result.detections.count) object(s) found")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Toggle("Masks", isOn: $showMask)
                    .fixedSize()
                Button("Clear") {
                    textResult = nil
                    selectedTextIndices.removeAll()
                    errorMessage = nil
                }
                .font(.caption)
            }
        }

        // Text input
        HStack(spacing: 8) {
            TextField("Type object name (e.g. dog, car)", text: $queryText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .submitLabel(.search)
                .onSubmit { runTextDetection() }

            Button(action: runTextDetection) {
                Image(systemName: "magnifyingglass")
                    .font(.body.bold())
            }
            .buttonStyle(.borderedProminent)
            .disabled(queryText.trimmingCharacters(in: .whitespaces).isEmpty || isProcessing)
        }

        if modelManager.textSession == nil {
            Text("Text detection models not available")
                .font(.caption)
                .foregroundColor(.orange)
        }
    }

    // MARK: - Gestures

    private func handleTap(at location: CGPoint, geometry: GeometryProxy) {
        guard inputMode == .point || inputMode == .both else { return }
        let imagePoint = viewToImage(location, viewSize: geometry.size)
        let point = SamPoint(
            x: imagePoint.x,
            y: imagePoint.y,
            label: showNegativePoints ? .negative : .positive
        )
        points.append(point)
        runSegmentation()
    }

    // MARK: - Segmentation

    private func runSegmentation() {
        guard !points.isEmpty || boundingBox != nil else { return }
        isProcessing = true
        errorMessage = nil

        Task {
            do {
                if modelManager.useSam2 {
                    try await runSam2Segmentation()
                } else {
                    try await runSamSegmentation()
                }
            } catch {
                await MainActor.run {
                    isProcessing = false
                    errorMessage = error.localizedDescription
                }
                print("Segmentation failed: \(error)")
            }
        }
    }

    private func runSamSegmentation() async throws {
        guard let session = modelManager.samSession else { return }

        if !samImageSet {
            try session.setImage(image.cgImage!)
            await MainActor.run { samImageSet = true }
        }

        let result = try session.predict(points: points, box: boundingBox)

        await MainActor.run {
            self.samResult = result
            self.selectedMaskIndex = 0
            self.isProcessing = false
        }
    }

    private func runSam2Segmentation() async throws {
        guard let session = modelManager.sam2Session else { return }

        if !sam2ImageSet {
            try session.setImage(image.cgImage!)
            await MainActor.run { sam2ImageSet = true }
        }

        let result = try await session.predict(points: points, box: boundingBox)

        await MainActor.run {
            self.samResult = result
            self.selectedMaskIndex = 0
            self.isProcessing = false
        }
    }

    private func runTextDetection() {
        let query = queryText.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty, !isProcessing else { return }
        guard let session = modelManager.textSession else {
            errorMessage = "Text detection models not loaded"
            return
        }

        isProcessing = true
        errorMessage = nil
        selectedTextIndices.removeAll()

        Task {
            do {
                if !textImageSet {
                    try session.setImage(image.cgImage!)
                    await MainActor.run { textImageSet = true }
                }

                let result = try session.segment(query: query)

                await MainActor.run {
                    self.textResult = result
                    self.isProcessing = false
                    if result.detections.isEmpty {
                        self.errorMessage = "No '\(query)' found in image"
                    }
                }
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    // MARK: - Image Setup

    private func setImageOnSessions() async {
        // Pre-cache image embeddings on available sessions
        if let session = modelManager.samSession, !samImageSet {
            do {
                try session.setImage(image.cgImage!)
                await MainActor.run { samImageSet = true }
            } catch {
                print("[UnifiedView] SAM setImage failed: \(error)")
            }
        }

        if let session = modelManager.textSession, !textImageSet {
            do {
                try session.setImage(image.cgImage!)
                await MainActor.run { textImageSet = true }
            } catch {
                print("[UnifiedView] Text setImage failed: \(error)")
            }
        }

        if let session = modelManager.sam2Session, !sam2ImageSet {
            do {
                try session.setImage(image.cgImage!)
                await MainActor.run { sam2ImageSet = true }
            } catch {
                print("[UnifiedView] SAM2 setImage failed: \(error)")
            }
        }
    }

    // MARK: - Actions

    private func clearAll() {
        points.removeAll()
        boundingBox = nil
        samResult = nil
        errorMessage = nil
    }

    private func clearModeResults() {
        // Clear results when switching modes, but keep cached image embeddings
        points.removeAll()
        boundingBox = nil
        samResult = nil
        textResult = nil
        selectedTextIndices.removeAll()
        selectedMaskIndex = 0
        errorMessage = nil
        showNegativePoints = false
    }

    private func toggleTextSelection(_ index: Int) {
        if selectedTextIndices.contains(index) {
            selectedTextIndices.remove(index)
        } else {
            selectedTextIndices.insert(index)
        }
    }

    // MARK: - Coordinate Conversion

    private func viewToImage(_ viewPoint: CGPoint, viewSize: CGSize) -> CGPoint {
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

        let x = (viewPoint.x - offset.x) / displayedSize.width * imageSize.width
        let y = (viewPoint.y - offset.y) / displayedSize.height * imageSize.height
        return CGPoint(
            x: min(max(x, 0), imageSize.width),
            y: min(max(y, 0), imageSize.height)
        )
    }

    private func imageToView(_ imagePoint: CGPoint, viewSize: CGSize) -> CGPoint {
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
}
