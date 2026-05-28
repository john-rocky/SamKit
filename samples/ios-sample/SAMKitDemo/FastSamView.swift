import SwiftUI
import UIKit
import SAMKit

/// FastSAM demo: one forward pass segments every object; tap to isolate one.
///
/// FastSAM is a YOLOv8-seg model rather than a SAM encoder/prompt-decoder, so the workflow
/// differs from `UnifiedSegmentationView`: the detector runs once in `setImage`, "everything"
/// mode shows the full instance map, and a tap simply *selects* the instance underneath.
struct FastSamView: View {
    let image: UIImage
    @EnvironmentObject var modelManager: ModelManager
    @Environment(\.dismiss) private var dismiss

    private enum Mode { case everything, tap }

    @State private var mode: Mode = .everything
    @State private var everythingOverlay: CGImage?
    @State private var selected: SamMask?
    @State private var imageSet = false
    @State private var isProcessing = false
    @State private var status: String = ""

    var body: some View {
        NavigationView {
            GeometryReader { geo in
                ZStack {
                    Color.black.ignoresSafeArea()

                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: geo.size.width, height: geo.size.height)
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            handleTap(at: location, geo: geo)
                        }

                    // Everything-mode instance map (semi-transparent overlay)
                    if mode == .everything, let overlay = everythingOverlay {
                        Image(uiImage: UIImage(cgImage: overlay))
                            .resizable()
                            .scaledToFit()
                            .frame(width: geo.size.width, height: geo.size.height)
                            .opacity(0.55)
                            .allowsHitTesting(false)
                    }

                    // Tap-selection: dim everything, brighten the chosen object
                    if mode == .tap, let mask = selected {
                        Color.black.opacity(0.45).allowsHitTesting(false)
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(width: geo.size.width, height: geo.size.height)
                            .mask(
                                Image(uiImage: UIImage(cgImage: mask.cgImage))
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: geo.size.width, height: geo.size.height)
                            )
                            .allowsHitTesting(false)
                    }

                    if isProcessing {
                        ProgressView().progressViewStyle(CircularProgressViewStyle(tint: .white)).scaleEffect(1.4)
                    }

                    VStack {
                        Spacer()
                        if !status.isEmpty {
                            Text(status)
                                .font(.caption).foregroundColor(.white)
                                .padding(.horizontal, 12).padding(.vertical, 6)
                                .background(Capsule().fill(Color.black.opacity(0.6)))
                                .padding(.bottom, 16)
                        }
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
                ToolbarItem(placement: .principal) {
                    Picker("Mode", selection: $mode) {
                        Text("Everything").tag(Mode.everything)
                        Text("Tap to pick").tag(Mode.tap)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 220)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button { selected = nil } label: { Image(systemName: "arrow.counterclockwise") }
                        .disabled(mode != .tap || selected == nil)
                }
            }
        }
        .task { await prepare() }
        .onChange(of: mode) { newMode in
            if newMode == .everything && everythingOverlay == nil { runEverything() }
        }
    }

    // MARK: - Inference

    private func prepare() async {
        guard let session = modelManager.fastSamSession else {
            status = "FastSAM model not loaded"
            return
        }
        guard !imageSet, let cg = image.cgImage else { return }
        isProcessing = true
        do {
            try session.setImage(cg)
            imageSet = true
            runEverything()
        } catch {
            status = error.localizedDescription
            isProcessing = false
        }
    }

    private func runEverything() {
        guard let session = modelManager.fastSamSession, imageSet else { return }
        isProcessing = true
        Task {
            let t0 = CFAbsoluteTimeGetCurrent()
            let overlay = try? session.segmentEverythingMask()
            let ms = Int((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            await MainActor.run {
                self.everythingOverlay = overlay
                self.isProcessing = false
                self.status = overlay == nil ? "No objects found" : "Segmented everything in \(ms)ms"
            }
        }
    }

    private func handleTap(at location: CGPoint, geo: GeometryProxy) {
        guard mode == .tap, let session = modelManager.fastSamSession, imageSet else { return }
        let p = viewToImage(location, viewSize: geo.size)
        isProcessing = true
        Task {
            let instance = try? session.segment(at: p)
            await MainActor.run {
                self.selected = instance?.mask
                self.isProcessing = false
                self.status = instance == nil ? "No object here" : "Score \(String(format: "%.2f", instance!.score))"
            }
        }
    }

    // MARK: - Coordinate mapping (view → original image), matching UnifiedSegmentationView

    private func viewToImage(_ viewPoint: CGPoint, viewSize: CGSize) -> CGPoint {
        let imageSize = image.size
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height

        let displayedSize: CGSize
        let offset: CGPoint
        if imageAspect > viewAspect {
            let w = viewSize.width, h = viewSize.width / imageAspect
            displayedSize = CGSize(width: w, height: h)
            offset = CGPoint(x: 0, y: (viewSize.height - h) / 2)
        } else {
            let h = viewSize.height, w = viewSize.height * imageAspect
            displayedSize = CGSize(width: w, height: h)
            offset = CGPoint(x: (viewSize.width - w) / 2, y: 0)
        }
        let x = (viewPoint.x - offset.x) / displayedSize.width * imageSize.width
        let y = (viewPoint.y - offset.y) / displayedSize.height * imageSize.height
        return CGPoint(x: min(max(x, 0), imageSize.width), y: min(max(y, 0), imageSize.height))
    }
}
