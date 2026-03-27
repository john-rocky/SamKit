import SwiftUI
import UIKit
import PhotosUI
import SAMKit
import SAMKitGrounding
import SAMKitUI

struct ContentView: View {
    @EnvironmentObject var modelManager: ModelManager
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var showSegmentationView = false

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
                        Toggle("Use HuggingFace SAM2 Models", isOn: $modelManager.useSam2)
                            .toggleStyle(SwitchToggleStyle())
                            .onChange(of: modelManager.useSam2) { useSam2 in
                                if useSam2 {
                                    modelManager.switchToSam2(type: modelManager.selectedSam2Model)
                                } else {
                                    modelManager.switchSamModel(to: modelManager.selectedModelType)
                                }
                            }

                        if modelManager.useSam2 {
                            Picker("SAM2 Model", selection: $modelManager.selectedSam2Model) {
                                Text("SAM2 Tiny (HF)").tag(Sam2ModelType.tiny)
                                Text("SAM2 Small (HF)").tag(Sam2ModelType.small)
                                Text("SAM2 Base (HF)").tag(Sam2ModelType.base)
                                Text("SAM2 Large (HF)").tag(Sam2ModelType.large)
                            }
                            .pickerStyle(MenuPickerStyle())
                            .background(Color(.systemGray6))
                            .cornerRadius(8)
                            .onChange(of: modelManager.selectedSam2Model) { type in
                                modelManager.switchToSam2(type: type)
                            }
                        } else {
                            Picker("Model", selection: $modelManager.selectedModelType) {
                                Text("MobileSAM (Fast)").tag(ModelType.mobileSam)
                                Text("SAM 2.1 Base").tag(ModelType.sam2_1_base)
                                Text("SAM 2.1 Large").tag(ModelType.sam2_1_large)
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            .onChange(of: modelManager.selectedModelType) { type in
                                modelManager.switchSamModel(to: type)
                            }
                        }
                    }
                }
                .padding(.horizontal)

                // Loading status
                if modelManager.isLoading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text(modelManager.loadingStatus)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                // Action Buttons
                VStack(spacing: 12) {
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

                    // Re-open segmentation button (for when user comes back)
                    if selectedImage != nil {
                        Button(action: {
                            showSegmentationView = true
                        }) {
                            Label("Segment", systemImage: "wand.and.stars")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                    }
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
                    UnifiedSegmentationView(image: image)
                        .environmentObject(modelManager)
                }
            }
            .onChange(of: selectedImage) { newImage in
                if newImage != nil {
                    showSegmentationView = true
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

// MARK: - UIImage Extension for Orientation Fix

extension UIImage {
    func fixedOrientation() -> UIImage {
        if imageOrientation == .up {
            return self
        }

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

        let cgImg = ctx.makeImage()!
        return UIImage(cgImage: cgImg)
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(ModelManager())
    }
}
