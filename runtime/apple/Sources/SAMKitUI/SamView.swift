import SwiftUI
import UIKit
import SAMKit
import UniformTypeIdentifiers
import Photos

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
                await MainActor.run { self.session = newSession }
            } catch {
                await MainActor.run { self.loadError = error }
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

    // Subject highlight
    @State private var binaryMask: CGImage?
    @State private var outlineImage: CGImage?

    // Lift state
    @State private var liftedImage: UIImage?
    @State private var isLifted = false
    @State private var liftDragOffset: CGSize = .zero
    @State private var liftStartTranslation: CGSize = .zero
    @State private var showLiftMenu = false
    @State private var toastMessage: String?
    @State private var showShareSheet = false

    // Unified gesture
    @State private var gestureStartTime: Date?
    @State private var lastGestureTranslation: CGSize = .zero

    private var currentMask: CGImage? {
        guard let r = result, r.masks.indices.contains(selectedMaskIndex) else { return nil }
        return r.masks[selectedMaskIndex].cgImage
    }

    var body: some View {
        VStack {
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
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                                            guard gestureStartTime != nil,
                                                  !isLifted,
                                                  binaryMask != nil else { return }
                                            let moved = hypot(lastGestureTranslation.width, lastGestureTranslation.height)
                                            guard moved < 15 else { return }
                                            liftStartTranslation = lastGestureTranslation
                                            handleLiftObject()
                                        }
                                    }
                                    lastGestureTranslation = value.translation

                                    if isLifted {
                                        liftDragOffset = CGSize(
                                            width: value.translation.width - liftStartTranslation.width,
                                            height: value.translation.height - liftStartTranslation.height
                                        )
                                    }
                                }
                                .onEnded { value in
                                    let elapsed = Date().timeIntervalSince(gestureStartTime ?? Date())
                                    let moved = hypot(value.translation.width, value.translation.height)
                                    gestureStartTime = nil
                                    lastGestureTranslation = .zero

                                    if isLifted {
                                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                            liftDragOffset = .zero
                                        }
                                        withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                            showLiftMenu = true
                                        }
                                        return
                                    }

                                    if elapsed < 0.3 && moved < 15 {
                                        let imagePoint = viewToImageCoordinates(
                                            value.startLocation, viewSize: geometry.size
                                        )
                                        addPoint(at: imagePoint)
                                    }
                                }
                        )

                    // Subject highlight
                    if let mask = binaryMask, !isLifted {
                        Color.black.opacity(0.25)
                            .allowsHitTesting(false)

                        Image(uiImage: image)
                            .resizable().scaledToFit()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .mask(
                                Image(uiImage: UIImage(cgImage: mask))
                                    .resizable().scaledToFit()
                                    .frame(width: geometry.size.width, height: geometry.size.height)
                            )
                            .allowsHitTesting(false)
                    }

                    // Glowing outline
                    if !isLifted, let outline = outlineImage {
                        GlowingOutlineView(outline: outline, width: geometry.size.width, height: geometry.size.height)
                    }

                    // Point markers
                    if !isLifted {
                        ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                            let viewPos = imageToViewCoordinates(
                                CGPoint(x: point.x, y: point.y), viewSize: geometry.size
                            )
                            Circle()
                                .fill(point.label == .positive ? Color.green : Color.red)
                                .frame(width: 12, height: 12)
                                .overlay(Circle().stroke(Color.white, lineWidth: 2))
                                .position(viewPos)
                        }
                    }

                    // Processing indicator
                    if isProcessing {
                        Color.black.opacity(0.3)
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)
                    }

                    // Subject lift overlay
                    if isLifted {
                        Color.black.opacity(0.4)
                            .ignoresSafeArea()
                            .allowsHitTesting(showLiftMenu)
                            .contentShape(Rectangle())
                            .onTapGesture { dismissLift() }

                        if let mask = binaryMask {
                            Image(uiImage: image)
                                .resizable().scaledToFit()
                                .frame(width: geometry.size.width, height: geometry.size.height)
                                .mask(
                                    Image(uiImage: UIImage(cgImage: mask))
                                        .resizable().scaledToFit()
                                        .frame(width: geometry.size.width, height: geometry.size.height)
                                )
                                .shadow(color: .black.opacity(0.6), radius: 24, y: 12)
                                .scaleEffect(showLiftMenu ? 1.0 : 1.05)
                                .offset(liftDragOffset)
                                .allowsHitTesting(false)
                                .animation(.spring(response: 0.35, dampingFraction: 0.75), value: showLiftMenu)
                        }

                        if showLiftMenu {
                            LiftContextMenuView(
                                onCopy: { performCopy() },
                                onSave: { performSave() },
                                onShare: { showShareSheet = true; dismissLift() }
                            )
                            .transition(.scale(scale: 0.8).combined(with: .opacity))
                        }
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
                        binaryMask = nil
                        outlineImage = nil
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
                        .onChange(of: selectedMaskIndex) { _ in updateMaskAndOutline() }
                    }
                }
            }
            .padding()
        }
        .overlay { ToastOverlay(message: toastMessage) }
        .sheet(isPresented: $showShareSheet) {
            if let lifted = liftedImage { ActivityViewController(items: [lifted]) }
        }
        .animation(.easeInOut(duration: 0.25), value: isLifted)
        .animation(.easeInOut(duration: 0.2), value: toastMessage != nil)
        .animation(.spring(response: 0.3, dampingFraction: 0.75), value: showLiftMenu)
    }

    // MARK: - Mask Processing

    private func updateMaskAndOutline() {
        guard let mask = currentMask else {
            binaryMask = nil
            outlineImage = nil
            return
        }
        Task {
            if let processed = processVisibleMasks([mask]) {
                await MainActor.run {
                    binaryMask = processed.binary
                    outlineImage = processed.outline
                }
            }
        }
    }

    // MARK: - Lift

    private func handleLiftObject() {
        guard let cgImage = image.cgImage,
              let r = result, r.masks.indices.contains(selectedMaskIndex) else { return }
        let mask = r.masks[selectedMaskIndex]

        guard let extracted = SamMask.extractObject(from: cgImage, masks: [mask]) else { return }
        liftedImage = UIImage(cgImage: extracted)

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

    private func performCopy() {
        guard let lifted = liftedImage else { return }
        let msg = copyObject(lifted)
        dismissLift()
        withAnimation { toastMessage = msg }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            withAnimation { toastMessage = nil }
        }
    }

    private func performSave() {
        guard let lifted = liftedImage else { return }
        dismissLift()
        saveObject(lifted) { msg in
            withAnimation { toastMessage = msg }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                withAnimation { toastMessage = nil }
            }
        }
    }

    // MARK: - Coordinate Conversion

    private func viewToImageCoordinates(_ viewPoint: CGPoint, viewSize: CGSize) -> CGPoint {
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
                    try session.setImage(image.cgImage!)
                }
                let newResult = try session.predict(points: points)
                await MainActor.run {
                    self.result = newResult
                    self.selectedMaskIndex = 0
                    self.isProcessing = false
                }
                updateMaskAndOutline()
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.errorMessage = error.localizedDescription
                }
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
