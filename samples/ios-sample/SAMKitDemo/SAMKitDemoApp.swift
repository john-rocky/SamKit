import SwiftUI

@main
struct SAMKitDemoApp: App {
    @StateObject private var modelManager = ModelManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelManager)
                .task { modelManager.loadModels() }
        }
    }
}
