import Foundation
import SAMKit
import SAMKitGrounding

/// Centralized model lifecycle manager.
/// Initializes SAM, CLIP, and YOLO-World models in the background at app startup.
@MainActor
final class ModelManager: ObservableObject {

    // MARK: - Published State

    @Published var samSession: SamSession?
    @Published var textSession: TextSegmentationSession?
    @Published var sam2Session: Sam2Session?

    @Published var isLoading = true
    @Published var loadingStatus = "Initializing..."
    @Published var error: Error?

    @Published var selectedModelType: ModelType = .mobileSam
    @Published var useSam2 = false

    // MARK: - Model Loading

    /// Load all models in the background. Call once at app launch.
    func loadModels() {
        guard isLoading else { return }

        let modelType = selectedModelType

        Task {
            // 1. Load SamSession (fastest, needed for point/box)
            loadingStatus = "Loading SAM..."
            do {
                let session = try await Self.createSamSession(modelType: modelType)
                self.samSession = session
            } catch {
                print("[ModelManager] SAM load failed: \(error)")
                self.error = error
            }

            // 2. Load TextSegmentationSession (YOLO-World + CLIP + SAM)
            loadingStatus = "Loading YOLO-World + CLIP..."
            do {
                let session = try await Self.createTextSession()
                self.textSession = session
            } catch {
                print("[ModelManager] Text session load failed: \(error)")
            }

            isLoading = false
            loadingStatus = "Ready"
        }
    }

    /// Switch to MobileSAM.
    func switchSamModel() {
        guard useSam2 || samSession == nil else { return }
        useSam2 = false

        samSession = nil
        sam2Session = nil

        Task {
            do {
                let session = try await Self.createSamSession(modelType: .mobileSam)
                self.samSession = session
            } catch {
                print("[ModelManager] SAM model switch failed: \(error)")
                self.error = error
            }
        }
    }

    /// Switch to SAM2 Tiny.
    func switchToSam2() {
        guard !useSam2 || sam2Session == nil else { return }
        useSam2 = true

        sam2Session = nil

        Task {
            do {
                let session = try await Self.createSam2Session(modelPrefix: "SAM2Tiny")
                self.sam2Session = session
            } catch {
                print("[ModelManager] SAM2 model switch failed: \(error)")
                self.error = error
            }
        }
    }

    /// Whether the active SAM session (standard or SAM2) is ready.
    var isSamReady: Bool {
        if useSam2 {
            return sam2Session != nil
        }
        return samSession != nil
    }

    /// Display name for the currently active model.
    var activeModelName: String {
        if useSam2 {
            return "SAM2 Tiny"
        }
        return "MobileSAM"
    }

    // MARK: - Factory helpers (nonisolated to run off main actor)

    private nonisolated static func createSamSession(modelType: ModelType) async throws -> SamSession {
        let samModel = try SamModelRef.bundled(modelType)
        let config = RuntimeConfig(computeUnits: .bestAvailable, enableFP16: true)
        return try SamSession(model: samModel, config: config)
    }

    private nonisolated static func createTextSession() async throws -> TextSegmentationSession {
        let groundingModel = try GroundingModelRef.bundled()
        let samModel = try SamModelRef.bundled(.mobileSam)
        let config = RuntimeConfig(computeUnits: .bestAvailable, enableFP16: true)
        return try TextSegmentationSession(
            groundingModel: groundingModel,
            samModel: samModel,
            config: config
        )
    }

    private nonisolated static func createSam2Session(modelPrefix: String) async throws -> Sam2Session {
        let config = RuntimeConfig(computeUnits: .neuralEnginePreferred, enableFP16: true)
        return try Sam2Session(modelName: modelPrefix, config: config)
    }
}
