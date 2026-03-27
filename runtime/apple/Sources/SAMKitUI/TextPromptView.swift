import SwiftUI
import UIKit
import SAMKit
import SAMKitGrounding

/// Interactive text-prompted segmentation view.
///
/// Users type object names (e.g. "dog", "car, person") and the view
/// displays detected bounding boxes with segmentation masks overlaid.
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

    public init(image: UIImage, session: TextSegmentationSession) {
        self.image = image
        self.session = session
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Image + overlays
            GeometryReader { geometry in
                ZStack {
                    // Base image
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geometry.size.width, height: geometry.size.height)

                    // Mask overlays
                    if showMasks, let result = result {
                        ForEach(Array(result.masks.enumerated()), id: \.offset) { index, mask in
                            let isSelected = selectedIndices.isEmpty || selectedIndices.contains(index)
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

                // Detection summary
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
                        .onSubmit {
                            runDetection()
                        }

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
        .onAppear {
            setImageIfNeeded()
        }
    }

    // MARK: - Actions

    private func setImageIfNeeded() {
        guard !imageSet else { return }
        Task {
            do {
                try session.setImage(image.cgImage!)
                await MainActor.run {
                    imageSet = true
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                }
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
            } catch {
                await MainActor.run {
                    self.isProcessing = false
                    self.errorMessage = error.localizedDescription
                }
            }
        }
    }

    private func toggleSelection(_ index: Int) {
        if selectedIndices.contains(index) {
            selectedIndices.remove(index)
        } else {
            selectedIndices.insert(index)
        }
    }

    // MARK: - Coordinate Conversion

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
