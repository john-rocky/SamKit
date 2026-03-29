import SwiftUI
import UIKit
import SAMKit
import SAMKitGrounding
import UniformTypeIdentifiers
import Photos

// MARK: - Tool Mode

enum ToolMode {
    case point
    case box
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
    @State private var toolMode: ToolMode = .point
    @State private var dragStart: CGPoint?
    @State private var dragEnd: CGPoint?
    @State private var showNegativePoints = false
    @State private var queryText: String = ""
    @State private var samImageSet = false
    @State private var sam2ImageSet = false
    @State private var textImageSet = false
    @State private var selectedTextIndices: Set<Int> = []
    @State private var errorMessage: String?
    @FocusState private var isTextFieldFocused: Bool

    // Lift object state
    @State private var liftedImage: UIImage?
    @State private var showLiftedObject = false
    @State private var toastMessage: String?
    @State private var showShareSheet = false

    var body: some View {
        NavigationView {
            GeometryReader { geometry in
                ZStack {
                    // Base image
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .contentShape(Rectangle())
                        .gesture(
                            toolMode == .box ?
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
                            if isTextFieldFocused {
                                isTextFieldFocused = false
                            } else {
                                handleTap(at: location, geometry: geometry)
                            }
                        }
                        .onLongPressGesture(minimumDuration: 0.5) {
                            handleLiftObject()
                        }

                    // SAM mask overlay (always show top mask)
                    if let result = samResult, !result.masks.isEmpty {
                        Image(uiImage: UIImage(cgImage: result.masks[0].cgImage))
                            .resizable()
                            .scaledToFit()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .opacity(0.6)
                            .allowsHitTesting(false)
                    }

                    // Text mask overlays
                    if let result = textResult {
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

                        // Text detection bounding boxes (hidden — masks only)
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
                    if let box = boundingBox {
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

                    // Processing indicator
                    if isProcessing {
                        Color.black.opacity(0.3)
                        VStack(spacing: 8) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(1.5)
                            Text("Processing...")
                                .foregroundColor(.white)
                                .font(.caption)
                        }
                    }

                    // Model not ready overlay
                    if !modelManager.isSamReady {
                        Color.black.opacity(0.5)
                        VStack {
                            ProgressView()
                            Text(modelManager.loadingStatus)
                                .foregroundColor(.white)
                                .font(.caption)
                        }
                    }

                    // Floating toolbar (top-left)
                    VStack {
                        floatingToolbar
                        Spacer()
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
                    .padding(.leading, 8)

                    // Error message (top, below toolbar)
                    if let error = errorMessage {
                        VStack {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(RoundedRectangle(cornerRadius: 8).fill(Color.red.opacity(0.8)))
                                .padding(.top, 56)
                            Spacer()
                        }
                        .frame(maxWidth: .infinity)
                        .allowsHitTesting(false)
                    }

                    // Text field overlay (bottom)
                    VStack {
                        Spacer()
                        textFieldOverlay
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
                }
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
        .overlay {
            if showLiftedObject, let lifted = liftedImage {
                liftedObjectOverlay(lifted)
                    .transition(.opacity)
            }
        }
        .overlay {
            if let message = toastMessage {
                VStack {
                    Spacer()
                    Text(message)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Capsule().fill(Color.black.opacity(0.75)))
                        .padding(.bottom, 100)
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .onAppear {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        withAnimation { toastMessage = nil }
                    }
                }
            }
        }
        .sheet(isPresented: $showShareSheet) {
            if let lifted = liftedImage {
                ActivityViewController(items: [lifted])
            }
        }
        .animation(.easeInOut(duration: 0.25), value: showLiftedObject)
        .animation(.easeInOut(duration: 0.2), value: toastMessage != nil)
    }

    // MARK: - Floating Toolbar

    @ViewBuilder
    private var floatingToolbar: some View {
        HStack(spacing: 6) {
            // Point mode
            Button {
                toolMode = .point
                showNegativePoints = false
            } label: {
                Image(systemName: "hand.point.up.left")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(toolMode == .point && !showNegativePoints ? .white : .primary)
                    .frame(width: 36, height: 36)
                    .background(toolMode == .point && !showNegativePoints ? Color.blue : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            // Negative point mode
            Button {
                toolMode = .point
                showNegativePoints = true
            } label: {
                Image(systemName: "minus.circle")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(toolMode == .point && showNegativePoints ? .white : .primary)
                    .frame(width: 36, height: 36)
                    .background(toolMode == .point && showNegativePoints ? Color.red : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            // Box mode
            Button {
                toolMode = .box
            } label: {
                Image(systemName: "rectangle.dashed")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(toolMode == .box ? .white : .primary)
                    .frame(width: 36, height: 36)
                    .background(toolMode == .box ? Color.blue : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            Divider().frame(height: 24)

            // Clear all
            Button {
                clearAll()
            } label: {
                Image(systemName: "trash")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.red)
                    .frame(width: 36, height: 36)
            }
            .disabled(points.isEmpty && boundingBox == nil && samResult == nil && textResult == nil)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.15), radius: 4, y: 2)
    }

    // MARK: - Text Field Overlay

    @ViewBuilder
    private var textFieldOverlay: some View {
        VStack(spacing: 4) {
            if let result = textResult, !result.detections.isEmpty {
                Text("\"\(queryText)\" — \(result.detections.count) object(s) found")
                    .font(.caption2)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.black.opacity(0.6)))
            }

            HStack(spacing: 8) {
                TextField("Search for objects...", text: $queryText)
                    .textFieldStyle(.plain)
                    .font(.subheadline)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.systemBackground).opacity(0.9))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .focused($isTextFieldFocused)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                    .submitLabel(.search)
                    .onSubmit { runTextDetection() }

                Button(action: runTextDetection) {
                    Image(systemName: "magnifyingglass")
                        .font(.body.bold())
                        .frame(width: 36, height: 36)
                        .background(Color.accentColor)
                        .foregroundColor(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .disabled(queryText.trimmingCharacters(in: .whitespaces).isEmpty || isProcessing)
            }

            if modelManager.textSession == nil {
                Text("Text detection not available")
                    .font(.caption2)
                    .foregroundColor(.orange)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.black.opacity(0.5)))
            }
        }
    }

    // MARK: - Gestures

    private func handleTap(at location: CGPoint, geometry: GeometryProxy) {
        guard toolMode == .point else { return }
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
            self.isProcessing = false
        }
    }

    private func runSam2Segmentation() async throws {
        guard let session = modelManager.sam2Session else { return }

        if !sam2ImageSet {
            try session.setImage(image.cgImage!)
            await MainActor.run { sam2ImageSet = true }
        }

        let result = try session.predict(points: points, box: boundingBox)

        await MainActor.run {
            self.samResult = result
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
        isTextFieldFocused = false

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
        textResult = nil
        selectedTextIndices.removeAll()
        errorMessage = nil
    }

    private func toggleTextSelection(_ index: Int) {
        if selectedTextIndices.contains(index) {
            selectedTextIndices.remove(index)
        } else {
            selectedTextIndices.insert(index)
        }
    }

    // MARK: - Object Lift

    @ViewBuilder
    private func liftedObjectOverlay(_ lifted: UIImage) -> some View {
        ZStack {
            Color.black.opacity(0.7)
                .ignoresSafeArea()
                .onTapGesture { dismissLift() }

            VStack(spacing: 0) {
                Spacer()

                Image(uiImage: lifted)
                    .resizable()
                    .scaledToFit()
                    .padding(32)
                    .background(
                        CheckerboardBackground()
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .padding(24)
                    )
                    .shadow(color: .black.opacity(0.5), radius: 24, y: 12)
                    .scaleEffect(showLiftedObject ? 1.0 : 0.9)

                Spacer()

                HStack(spacing: 40) {
                    Button { copyObject() } label: {
                        VStack(spacing: 6) {
                            Image(systemName: "doc.on.doc")
                                .font(.title2)
                            Text("Copy")
                                .font(.caption)
                        }
                    }
                    Button { saveObject() } label: {
                        VStack(spacing: 6) {
                            Image(systemName: "square.and.arrow.down")
                                .font(.title2)
                            Text("Save")
                                .font(.caption)
                        }
                    }
                    Button { shareObject() } label: {
                        VStack(spacing: 6) {
                            Image(systemName: "square.and.arrow.up")
                                .font(.title2)
                            Text("Share")
                                .font(.caption)
                        }
                    }
                }
                .foregroundColor(.white)
                .padding(.bottom, 50)
            }
        }
    }

    private func handleLiftObject() {
        guard let cgImage = image.cgImage else { return }

        let visibleMasks: [SamMask]
        if let result = samResult, !result.masks.isEmpty {
            visibleMasks = [result.masks[0]]
        } else if let result = textResult, !result.masks.isEmpty {
            let indices = selectedTextIndices.isEmpty
                ? Set(0..<result.masks.count)
                : selectedTextIndices
            visibleMasks = indices.sorted().compactMap { $0 < result.masks.count ? result.masks[$0] : nil }
        } else {
            visibleMasks = []
        }

        guard !visibleMasks.isEmpty,
              let extracted = SamMask.extractObject(from: cgImage, masks: visibleMasks) else { return }

        liftedImage = UIImage(cgImage: extracted)
        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            showLiftedObject = true
        }
    }

    private func dismissLift() {
        withAnimation(.easeOut(duration: 0.2)) {
            showLiftedObject = false
        }
    }

    private func copyObject() {
        guard let lifted = liftedImage,
              let pngData = lifted.pngData() else { return }
        UIPasteboard.general.setData(pngData, forPasteboardType: UTType.png.identifier)
        withAnimation { toastMessage = "Copied to clipboard" }
        dismissLift()
    }

    private func saveObject() {
        guard let lifted = liftedImage,
              let pngData = lifted.pngData() else { return }

        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    withAnimation { toastMessage = "Photo access denied" }
                    dismissLift()
                }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: pngData, options: nil)
            } completionHandler: { success, _ in
                DispatchQueue.main.async {
                    withAnimation { toastMessage = success ? "Saved to Photos" : "Save failed" }
                    dismissLift()
                }
            }
        }
    }

    private func shareObject() {
        showShareSheet = true
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

// MARK: - Checkerboard Background (indicates transparency)

private struct CheckerboardBackground: View {
    let tileSize: CGFloat = 10

    var body: some View {
        Canvas { context, size in
            let cols = Int(ceil(size.width / tileSize))
            let rows = Int(ceil(size.height / tileSize))
            for row in 0..<rows {
                for col in 0..<cols {
                    let isLight = (row + col) % 2 == 0
                    let rect = CGRect(
                        x: CGFloat(col) * tileSize,
                        y: CGFloat(row) * tileSize,
                        width: tileSize,
                        height: tileSize
                    )
                    context.fill(Path(rect), with: .color(isLight ? .white : Color(white: 0.85)))
                }
            }
        }
    }
}

// MARK: - Share Sheet

private struct ActivityViewController: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
