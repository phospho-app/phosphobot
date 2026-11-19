import Darwin
import Foundation
import Network

// The test process impersonates a ready server while its newly launched child sleeps.
@main
struct BackendPortRace {
    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ description: String) { errorDescription = description }
    }

    static func main() throws {
        if CommandLine.arguments.dropFirst().first == "run" {
            sleep(30)
            return
        }

        let log = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: log) }
        var state: BackendServer.State = .stopped
        let backend = BackendServer(logURL: log) { state = $0 }
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        backend.start(executable: executable, simulationOnly: true)
        defer { backend.stop() }
        guard case .starting(let url) = state, let port = url.port else {
            throw Failure("Backend did not enter startup")
        }

        let queue = DispatchQueue(label: "port-race-listener")
        let tcpPort = NWEndpoint.Port(rawValue: UInt16(port))!
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: tcpPort)
        let other = try NWListener(using: parameters)
        defer { other.cancel() }
        let body = #"{"status":"ok","server_port":\#(port)}"#
        let response = "HTTP/1.1 200 OK\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n" + body
        other.newConnectionHandler = { connection in
            connection.start(queue: queue)
            connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { _, _, _, _ in
                connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in connection.cancel() })
            }
        }
        other.start(queue: queue)

        let deadline = Date().addingTimeInterval(12)
        while Date() < deadline {
            switch state {
            case .running:
                throw Failure("Launcher accepted a healthy response from an unrelated process")
            case .failed(let reason):
                guard reason.contains("Another service took port \(port)") else { throw Failure(reason) }
                let surviving = try Data(contentsOf: url.appendingPathComponent("status"))
                guard String(decoding: surviving, as: UTF8.self) == body else {
                    throw Failure("The unrelated listener did not survive")
                }
                print("OK: a competing listener was identified and kept running")
                return
            default:
                RunLoop.main.run(until: Date().addingTimeInterval(0.05))
            }
        }
        throw Failure("Timed out waiting for the port conflict")
    }
}
