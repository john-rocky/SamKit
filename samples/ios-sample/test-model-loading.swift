#!/usr/bin/env swift

// Simple test script to verify SAM2 model loading
import Foundation

#if canImport(UIKit)
import UIKit
#endif

// Test function to check if models can be found
func testModelAvailability() {
    let bundle = Bundle.main
    let modelTypes = ["SAM2TinyImageEncoderFLOAT16", "SAM2TinyPromptEncoderFLOAT16", "SAM2TinyMaskDecoderFLOAT16"]
    
    print("🔍 Checking SAM2 model availability...")
    
    for modelName in modelTypes {
        if let url = bundle.url(forResource: modelName, withExtension: "mlmodelc") {
            print("✅ Found: \(modelName) at \(url.path)")
        } else {
            print("❌ Missing: \(modelName)")
            
            // Check if .mlpackage exists instead
            if let packageUrl = bundle.url(forResource: modelName, withExtension: "mlpackage") {
                print("⚠️  Found .mlpackage but not compiled: \(packageUrl.path)")
            }
        }
    }
    
    // List all .mlmodelc files in bundle
    if let bundlePath = bundle.resourcePath {
        print("\n📁 All .mlmodelc files in bundle:")
        let fileManager = FileManager.default
        do {
            let files = try fileManager.contentsOfDirectory(atPath: bundlePath)
            for file in files.sorted() {
                if file.hasSuffix(".mlmodelc") {
                    print("   - \(file)")
                }
            }
        } catch {
            print("Error listing bundle contents: \(error)")
        }
    }
}

print("📱 SAM2 Model Loading Test")
print("=" * 40)

testModelAvailability()

print("\n✨ Test completed!")