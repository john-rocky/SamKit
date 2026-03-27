import SwiftUI
import SAMKit
import SAMKitGrounding
import SAMKitUI

/// Full-screen text-prompted segmentation view for the demo app.
struct TextSegmentationView: View {
    let image: UIImage
    let samModelType: ModelType
    @Environment(\.dismiss) private var dismiss
    @State private var session: TextSegmentationSession?
    @State private var isLoading = true
    @State private var loadError: Error?

    var body: some View {
        NavigationView {
            ZStack {
                if let session = session {
                    TextPromptView(image: image, session: session)
                } else if isLoading {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Loading models...")
                            .padding(.top)
                        Text("YOLO-World + \(samModelType.displayName)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } else if let error = loadError {
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 50))
                            .foregroundColor(.red)
                        Text("Failed to load models")
                            .font(.headline)
                        Text(error.localizedDescription)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        Text("Make sure grounding model files (.mlpackage) and clip_vocab.json are added to the project")
                            .font(.caption)
                            .foregroundColor(.orange)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Back") { dismiss() }
                }
                ToolbarItem(placement: .principal) {
                    Text("Text Prompt")
                        .font(.headline)
                }
            }
        }
        .onAppear { loadModels() }
    }

    private func loadModels() {
        Task {
            do {
                let groundingModel = try GroundingModelRef.bundled()
                let samModel = try SamModelRef.bundled(samModelType)

                let config = RuntimeConfig(
                    computeUnits: .bestAvailable,
                    enableFP16: true
                )

                let newSession = try TextSegmentationSession(
                    groundingModel: groundingModel,
                    samModel: samModel,
                    config: config
                )

                await MainActor.run {
                    self.session = newSession
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
