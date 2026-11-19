import Foundation

// Run with the official macOS binary as the first argument. See README.md.
@main
struct BackendSmoke {
    struct Failure: Error, CustomStringConvertible {
        var description: String
        init(_ description: String) { self.description = description }
    }

    static func wait(seconds: TimeInterval, until condition: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() && Date() < deadline {
            RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        }
        return condition()
    }

    static func fetch(_ url: URL) throws -> Data {
        let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 6)
        let done = DispatchSemaphore(value: 0)
        var bytes: Data?
        URLSession.shared.dataTask(with: request) { data, response, _ in
            if (response as? HTTPURLResponse)?.statusCode == 200 { bytes = data }
            done.signal()
        }.resume()
        guard done.wait(timeout: .now()+8) == .success, let bytes, !bytes.isEmpty else {
            throw Failure("Expected HTTP 200 at \(url.path)")
        }
        return bytes
    }

    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw Failure("Pass the path to a phosphobot release executable") }
        let log = FileManager.default.temporaryDirectory.appendingPathComponent("phosphobot-smoke-\(UUID().uuidString).log")
        var current: BackendServer.State = .stopped
        let server = BackendServer(logURL: log) { state in current = state }
        let executable = URL(fileURLWithPath: CommandLine.arguments[1])
        server.start(executable: executable, simulationOnly: true)
        defer {
            server.stop()
            _ = wait(seconds: 15) { if case .stopping = current { return false }; return true }
        }
        var address: URL?
        var errorMessage: String?
        _ = wait(seconds: 250) {
            switch current {
            case .running(let url): address = url; return true
            case .failed(let message): errorMessage = message; return true
            default: return false
            }
        }
        guard let address else { throw Failure("\(errorMessage ?? "Timed out during backend startup"). See \(log.path)") }
        let page = String(decoding: try fetch(address), as: UTF8.self)
        guard page.contains("<script"), page.contains("<div") else { throw Failure("Dashboard missing its application shell") }
        let docs = String(decoding: try fetch(address.appendingPathComponent("docs")), as: UTF8.self)
        guard docs.contains("SwaggerUIBundle") else { throw Failure("Missing Swagger UI") }
        _ = try fetch(address.appendingPathComponent("resources/swagger-ui/swagger-ui-bundle.js"))
        let schema = try JSONSerialization.jsonObject(with: fetch(address.appendingPathComponent("openapi.json"))) as? [String: Any]
        guard let paths = schema?["paths"] as? [String: Any], paths["/status"] != nil else {
            throw Failure("Missing server API routes")
        }
        print("OK: real dashboard, Swagger UI and local JS, OpenAPI (\(paths.count) paths)")
        server.stop()
        guard wait(seconds: 20, until: { if case .stopped = current { return true }; return false }) else {
            throw Failure("The backend did not shut down. See \(log.path)")
        }
        if (try? fetch(address.appendingPathComponent("status"))) != nil { throw Failure("Backend still accepts requests after Stop") }
        try? FileManager.default.removeItem(at: log)
        print("OK: Stop closed the server port")
    }
}
