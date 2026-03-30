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

    // Processed mask and outline
    @State private var binaryMask: CGImage?
    @State private var outlineImage: CGImage?

    // Lift object state
    @State private var liftedImage: UIImage?
    @State private var isLifted = false
    @State private var liftDragOffset: CGSize = .zero
    @State private var liftStartTranslation: CGSize = .zero
    @State private var showLiftMenu = false
    @State private var toastMessage: String?
    @State private var showShareSheet = false

    // Unified gesture tracking
    @State private var gestureStartTime: Date?
    @State private var lastGestureTranslation: CGSize = .zero

    private var hasVisibleMasks: Bool {
        if let result = samResult, !result.masks.isEmpty { return true }
        if let result = textResult, !result.masks.isEmpty { return true }
        return false
    }

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
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    if gestureStartTime == nil {
                                        gestureStartTime = Date()
                                        // Schedule long press check via timer
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                            guard gestureStartTime != nil,
                                                  !isLifted,
                                                  hasVisibleMasks,
                                                  dragStart == nil else { return }
                                            let moved = hypot(lastGestureTranslation.width, lastGestureTranslation.height)
                                            guard moved < 15 else { return }
                                            liftStartTranslation = lastGestureTranslation
                                            handleLiftObject()
                                        }
                                    }

                                    lastGestureTranslation = value.translation

                                    // Already lifted — track drag offset
                                    if isLifted {
                                        liftDragOffset = CGSize(
                                            width: value.translation.width - liftStartTranslation.width,
                                            height: value.translation.height - liftStartTranslation.height
                                        )
                                        return
                                    }

                                    // Box drag
                                    let moved = hypot(value.translation.width, value.translation.height)
                                    if toolMode == .box && moved >= 10 {
                                        if dragStart == nil { dragStart = value.startLocation }
                                        dragEnd = value.location
                                    }
                                }
                                .onEnded { value in
                                    let elapsed = Date().timeIntervalSince(gestureStartTime ?? Date())
                                    let moved = hypot(value.translation.width, value.translation.height)
                                    gestureStartTime = nil
                                    lastGestureTranslation = .zero

                                    // Lifted — snap back and show menu
                                    if isLifted {
                                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                            liftDragOffset = .zero
                                        }
                                        withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                            showLiftMenu = true
                                        }
                                        return
                                    }

                                    // Box drag ended
                                    if toolMode == .box, let start = dragStart {
                                        let startPt = viewToImage(start, viewSize: geometry.size)
                                        let endPt = viewToImage(value.location, viewSize: geometry.size)
                                        boundingBox = SamBox(
                                            x0: Float(min(startPt.x, endPt.x)),
                                            y0: Float(min(startPt.y, endPt.y)),
                                            x1: Float(max(startPt.x, endPt.x)),
                                            y1: Float(max(startPt.y, endPt.y))
                                        )
                                        dragStart = nil
                                        dragEnd = nil
                                        runSegmentation()
                                        return
                                    }

                                    // Quick tap — add point or dismiss keyboard
                                    if elapsed < 0.3 && moved < 15 {
                                        if isTextFieldFocused {
                                            isTextFieldFocused = false
                                        } else {
                                            handleTap(at: value.startLocation, geometry: geometry)
                                        }
                                    }
                                }
                        )

                    // Subject highlight — dim background + bright subject
                    if hasVisibleMasks && !isLifted {
                        Color.black.opacity(0.25)
                            .allowsHitTesting(false)

                        liftedSubjectView(geometry: geometry)
                            .allowsHitTesting(false)
                    }

                    // Animated glowing outline
                    if !isLifted, let outline = outlineImage {
                        glowingOutlineView(outline: outline, geometry: geometry)
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

                    // Point markers — hide when lifted
                    if !isLifted {
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

                    // Floating toolbar — hide when lifted
                    if !isLifted {
                        VStack {
                            floatingToolbar
                            Spacer()
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 8)
                        .padding(.leading, 8)
                    }

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

                    // Text field overlay — hide when lifted
                    if !isLifted {
                        VStack {
                            Spacer()
                            textFieldOverlay
                        }
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)
                    }

                    // MARK: In-place subject lift
                    if isLifted {
                        // Dimmed background — interactive only when menu is showing
                        Color.black.opacity(0.4)
                            .ignoresSafeArea()
                            .allowsHitTesting(showLiftMenu)
                            .contentShape(Rectangle())
                            .onTapGesture { dismissLift() }

                        // Lifted object — visual only, drag handled by base gesture
                        liftedSubjectView(geometry: geometry)
                            .shadow(color: .black.opacity(0.6), radius: 24, y: 12)
                            .scaleEffect(showLiftMenu ? 1.0 : 1.05)
                            .offset(liftDragOffset)
                            .allowsHitTesting(false)
                            .animation(.spring(response: 0.35, dampingFraction: 0.75), value: showLiftMenu)

                        // Context menu
                        if showLiftMenu {
                            liftContextMenu
                                .transition(.scale(scale: 0.8).combined(with: .opacity))
                        }
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") {
                        if isLifted { dismissLift() }
                        else { dismiss() }
                    }
                }
                ToolbarItem(placement: .principal) {
                    Text(modelManager.activeModelName)
                        .font(.headline)
                }
            }
        }
        .task { await setImageOnSessions() }
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
        .animation(.easeInOut(duration: 0.25), value: isLifted)
        .animation(.easeInOut(duration: 0.2), value: toastMessage != nil)
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: showLiftMenu)
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
        updateOutline()
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
        updateOutline()
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
                updateOutline()
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
        binaryMask = nil
        outlineImage = nil
    }

    private func toggleTextSelection(_ index: Int) {
        if selectedTextIndices.contains(index) {
            selectedTextIndices.remove(index)
        } else {
            selectedTextIndices.insert(index)
        }
    }

    // MARK: - Subject Lift

    @ViewBuilder
    private func liftedSubjectView(geometry: GeometryProxy) -> some View {
        Image(uiImage: image)
            .resizable()
            .scaledToFit()
            .frame(width: geometry.size.width, height: geometry.size.height)
            .mask(
                maskOverlayView(geometry: geometry)
            )
    }

    @ViewBuilder
    private func maskOverlayView(geometry: GeometryProxy) -> some View {
        if let mask = binaryMask {
            Image(uiImage: UIImage(cgImage: mask))
                .resizable()
                .scaledToFit()
                .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }

    @ViewBuilder
    private var liftContextMenu: some View {
        VStack(spacing: 0) {
            Button {
                copyObject()
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .contentShape(Rectangle())
            }
            Divider()
            Button {
                saveObject()
            } label: {
                Label("Save to Photos", systemImage: "square.and.arrow.down")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .contentShape(Rectangle())
            }
            Divider()
            Button {
                shareObject()
                dismissLift()
            } label: {
                Label("Share...", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .contentShape(Rectangle())
            }
        }
        .foregroundColor(.primary)
        .font(.body)
        .frame(width: 220)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .shadow(color: .black.opacity(0.2), radius: 20, y: 5)
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

        // Cropped image for copy/save/share actions
        liftedImage = UIImage(cgImage: extracted)

        // Haptic feedback
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()

        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            isLifted = true
        }
    }

    private func dismissLift() {
        withAnimation(.easeOut(duration: 0.2)) {
            showLiftMenu = false
            isLifted = false
        }
        liftDragOffset = .zero
        liftStartTranslation = .zero
    }

    private func copyObject() {
        guard let lifted = liftedImage,
              let pngData = lifted.pngData() else { return }
        UIPasteboard.general.setData(pngData, forPasteboardType: UTType.png.identifier)
        dismissLift()
        withAnimation { toastMessage = "Copied to clipboard" }
    }

    private func saveObject() {
        guard let lifted = liftedImage,
              let pngData = lifted.pngData() else { return }
        dismissLift()

        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    withAnimation { toastMessage = "Photo access denied" }
                }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: pngData, options: nil)
            } completionHandler: { success, _ in
                DispatchQueue.main.async {
                    withAnimation { toastMessage = success ? "Saved to Photos" : "Save failed" }
                }
            }
        }
    }

    private func shareObject() {
        showShareSheet = true
    }

    // MARK: - Glowing Outline

    @ViewBuilder
    private func glowingOutlineView(outline: CGImage, geometry: GeometryProxy) -> some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30)) { timeline in
            let t = timeline.date.timeIntervalSinceReferenceDate
            let phase = t.truncatingRemainder(dividingBy: 2.5) / 2.5

            ZStack {
                // Soft glow layer
                Image(uiImage: UIImage(cgImage: outline))
                    .resizable()
                    .scaledToFit()
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .colorMultiply(Color(red: 0.5, green: 0.85, blue: 1.0))
                    .blur(radius: 5)
                    .opacity(0.8)

                // Sharp outline
                Image(uiImage: UIImage(cgImage: outline))
                    .resizable()
                    .scaledToFit()
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .colorMultiply(.white)

                // Traveling shimmer highlight
                Image(uiImage: UIImage(cgImage: outline))
                    .resizable()
                    .scaledToFit()
                    .frame(width: geometry.size.width, height: geometry.size.height)
                    .colorMultiply(.white)
                    .mask(
                        AngularGradient(
                            gradient: Gradient(colors: [
                                .white, .white.opacity(0.5), .clear, .clear,
                                .clear, .clear, .clear, .white.opacity(0.3)
                            ]),
                            center: .center,
                            startAngle: .degrees(phase * 360),
                            endAngle: .degrees(phase * 360 + 360)
                        )
                    )
                    .blur(radius: 2)
            }
            .allowsHitTesting(false)
        }
    }

    private func generateOutline(from maskImage: CGImage) -> CGImage? {
        let width = maskImage.width
        let height = maskImage.height
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        let glowRadius = CGFloat(min(30, max(4, min(width, height) / 100)))

        // Step 1: Create white silhouette from mask alpha
        guard let silCtx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        silCtx.draw(maskImage, in: rect)
        silCtx.setBlendMode(.sourceIn)
        silCtx.setFillColor(UIColor.white.cgColor)
        silCtx.fill(rect)
        guard let whiteSilhouette = silCtx.makeImage() else { return nil }

        // Step 2: Draw silhouette with shadow glow, then erase interior
        guard let outCtx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        outCtx.setShadow(offset: .zero, blur: glowRadius, color: UIColor.white.cgColor)
        outCtx.draw(whiteSilhouette, in: rect)

        outCtx.setShadow(offset: .zero, blur: 0, color: nil)
        outCtx.setBlendMode(.destinationOut)
        outCtx.draw(whiteSilhouette, in: rect)

        return outCtx.makeImage()
    }

    private func composeMasks(_ masks: [CGImage]) -> CGImage? {
        guard let first = masks.first else { return nil }
        if masks.count == 1 { return first }

        let width = first.width
        let height = first.height
        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        for mask in masks {
            ctx.draw(mask, in: rect)
        }
        return ctx.makeImage()
    }

    private func binarizeMask(_ maskImage: CGImage) -> CGImage? {
        let width = maskImage.width
        let height = maskImage.height

        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        ctx.draw(maskImage, in: rect)

        guard let data = ctx.data else { return nil }
        let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        let threshold: UInt8 = 128
        for i in 0..<(width * height) {
            let o = i * 4
            if pixels[o + 3] >= threshold {
                pixels[o] = 255
                pixels[o + 1] = 255
                pixels[o + 2] = 255
                pixels[o + 3] = 255
            } else {
                pixels[o] = 0
                pixels[o + 1] = 0
                pixels[o + 2] = 0
                pixels[o + 3] = 0
            }
        }

        return ctx.makeImage()
    }

    private func updateOutline() {
        let currentSamResult = samResult
        let currentTextResult = textResult
        let currentIndices = selectedTextIndices

        Task {
            var maskImages: [CGImage] = []
            if let result = currentSamResult, !result.masks.isEmpty {
                maskImages.append(result.masks[0].cgImage)
            }
            if let result = currentTextResult, !result.masks.isEmpty {
                let indices = currentIndices.isEmpty
                    ? Set(0..<result.masks.count)
                    : currentIndices
                maskImages += indices.sorted().compactMap {
                    $0 < result.masks.count ? result.masks[$0].cgImage : nil
                }
            }

            guard !maskImages.isEmpty else {
                await MainActor.run {
                    binaryMask = nil
                    outlineImage = nil
                }
                return
            }

            let composed = composeMasks(maskImages)
            let binary = composed.flatMap { binarizeMask($0) }
            let outline = binary.flatMap { generateOutline(from: $0) }

            await MainActor.run {
                self.binaryMask = binary
                self.outlineImage = outline
            }
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

// MARK: - Share Sheet

private struct ActivityViewController: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
