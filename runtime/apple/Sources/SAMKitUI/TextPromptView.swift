import SwiftUI
import UIKit
import SAMKit
import SAMKitGrounding
import UniformTypeIdentifiers
import Photos

public struct TextPromptView: View {
    let image: UIImage
    let session: TextSegmentationSession

    @State private var queryText: String = ""
    @State private var result: TextSegmentationResult?
    @State private var isProcessing = false
    @State private var imageSet = false
    @State private var errorMessage: String?
    @State private var selectedIndices: Set<Int> = []
    @State private var showMasks = true

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

    public init(image: UIImage, session: TextSegmentationSession) {
        self.image = image
        self.session = session
    }

    private var visibleMaskImages: [CGImage] {
        guard let result = result, !result.masks.isEmpty else { return [] }
        let indices = selectedIndices.isEmpty
            ? Set(0..<result.masks.count)
            : selectedIndices
        return indices.sorted().compactMap {
            $0 < result.masks.count ? result.masks[$0].cgImage : nil
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
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
                                    gestureStartTime = nil
                                    lastGestureTranslation = .zero

                                    if isLifted {
                                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                                            liftDragOffset = .zero
                                        }
                                        withAnimation(.spring(response: 0.25, dampingFraction: 0.8)) {
                                            showLiftMenu = true
                                        }
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

                    // Processing indicator
                    if isProcessing {
                        Color.black.opacity(0.3)
                        VStack(spacing: 8) {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(1.5)
                            Text("Detecting...")
                                .foregroundColor(.white)
                                .font(.caption)
                        }
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
            VStack(spacing: 12) {
                if let error = errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                        .lineLimit(2)
                }

                if let result = result, !result.detections.isEmpty {
                    HStack {
                        Text("\(result.detections.count) object(s) found")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Toggle("Masks", isOn: $showMasks)
                            .fixedSize()
                        Button("Clear") {
                            self.result = nil
                            selectedIndices.removeAll()
                            binaryMask = nil
                            outlineImage = nil
                            errorMessage = nil
                        }
                        .font(.caption)
                    }
                }

                HStack(spacing: 8) {
                    TextField("Type object name (e.g. dog, car)", text: $queryText)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .submitLabel(.search)
                        .onSubmit { runDetection() }

                    Button(action: runDetection) {
                        Image(systemName: "magnifyingglass")
                            .font(.body.bold())
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(queryText.trimmingCharacters(in: .whitespaces).isEmpty || isProcessing)
                }
            }
            .padding()
        }
        .onAppear { setImageIfNeeded() }
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
        let masks = visibleMaskImages
        guard !masks.isEmpty else {
            binaryMask = nil
            outlineImage = nil
            return
        }
        Task {
            if let processed = processVisibleMasks(masks) {
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
              let result = result, !result.masks.isEmpty else { return }

        let indices = selectedIndices.isEmpty
            ? Set(0..<result.masks.count)
            : selectedIndices
        let masks = indices.sorted().compactMap {
            $0 < result.masks.count ? result.masks[$0] : nil
        }
        guard !masks.isEmpty,
              let extracted = SamMask.extractObject(from: cgImage, masks: masks) else { return }

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

    // MARK: - Actions

    private func setImageIfNeeded() {
        guard !imageSet else { return }
        Task {
            do {
                try session.setImage(image.cgImage!)
                await MainActor.run { imageSet = true }
            } catch {
                await MainActor.run { errorMessage = error.localizedDescription }
            }
        }
    }

    private func runDetection() {
        let query = queryText.trimmingCharacters(in: .whitespaces)
        guard !query.isEmpty, !isProcessing else { return }

        isProcessing = true
        errorMessage = nil
        selectedIndices.removeAll()

        Task {
            do {
                if !imageSet {
                    try session.setImage(image.cgImage!)
                    await MainActor.run { imageSet = true }
                }

                let newResult = try session.segment(query: query)

                await MainActor.run {
                    self.result = newResult
                    self.isProcessing = false
                    if newResult.detections.isEmpty {
                        self.errorMessage = "No '\(query)' found in image"
                    }
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
