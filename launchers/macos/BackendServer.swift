import Darwin
import Foundation

/// Owns the one backend launched by this app. Call on the main thread.
final class BackendServer {
    enum State {
        case stopped, starting(URL), running(URL), stopping, failed(String)
    }

    let logURL: URL
    private let onChange: (State) -> Void
    private(set) var state: State = .stopped {
        didSet { onChange(state) }
    }

    private var process: Process?
    private var ownsGroup = false
    private var log: FileHandle?
    private var healthCheck: URLSessionDataTask?
    private var monitor: Timer?
    private var stopReason: String?
    private let session: URLSession

    init(logURL: URL, onChange: @escaping (State) -> Void) {
        self.logURL = logURL
        self.onChange = onChange
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        configuration.timeoutIntervalForRequest = 2
        configuration.timeoutIntervalForResource = 3
        session = URLSession(configuration: configuration)
    }

    private static func availablePort() throws -> Int {
        for port in 8020...8039 {
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { throw POSIXError(.ENFILE) }
            defer { close(fd) }
            var addr = sockaddr_in()
            addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = in_port_t(port).bigEndian
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")
            let bound = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            if bound == 0 { return port }
        }
        throw NSError(domain: "phosphobot", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Local ports 8020 through 8039 are busy. Stop one of the services and try again."
        ])
    }

    func start(executable: URL, simulationOnly: Bool) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard process == nil else { return }
        do {
            let port = try Self.availablePort()
            let url = URL(string: "http://127.0.0.1:\(port)")!
            let fm = FileManager.default
            try fm.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Start each session with fresh diagnostics; upstream also rotates its own logs.
            try Data().write(to: logURL, options: .atomic)
            let handle = try FileHandle(forWritingTo: logURL)
            log = handle
            let child = Process()
            child.executableURL = executable
            child.currentDirectoryURL = fm.homeDirectoryForCurrentUser
            child.arguments = ["run", "--host", "127.0.0.1", "--port", String(port)]
            if simulationOnly {
                child.arguments! += ["--only-simulation", "--no-cameras", "--no-realsense", "--no-can"]
            }
            child.standardInput = FileHandle.nullDevice
            child.standardOutput = handle
            child.standardError = handle
            child.terminationHandler = { [weak self] finished in
                DispatchQueue.main.async { self?.finished(finished) }
            }
            try child.run()
            process = child
            // macOS Foundation launches its child in a new process group. Record and
            // verify this before ever signalling a group (the one-file release forks).
            ownsGroup = getpgid(child.processIdentifier) == child.processIdentifier
            state = .starting(url)
            let deadline = Date().addingTimeInterval(240)
            monitor = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { [weak self] _ in
                self?.check(child, url: url, deadline: deadline)
            }
        } catch {
            try? log?.close()
            log = nil
            state = .failed(error.localizedDescription)
        }
    }

    private enum ListenerOwner { case server, other, unknown }

    private static func listenerOwner(port: Int, childPID: pid_t, ownsGroup: Bool) -> ListenerOwner {
        // The PyInstaller bootloader can fork the actual listener. Inspect listening
        // sockets rather than trusting an HTTP response from the selected port.
        let inspector = Process()
        let output = Pipe()
        inspector.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        inspector.arguments = ["-nP", "-a", "-iTCP:\(port)", "-sTCP:LISTEN", "-Fpn"]
        inspector.standardOutput = output
        inspector.standardError = FileHandle.nullDevice
        do { try inspector.run() } catch { return .unknown }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        inspector.waitUntilExit()
        guard inspector.terminationStatus == 0 else { return .unknown }

        var pid: pid_t?
        var owners = Set<pid_t>()
        for field in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            if field.first == "p" { pid = pid_t(field.dropFirst()) }
            // Wildcard listeners can also answer requests sent to IPv4 loopback.
            if let pid, field == "n127.0.0.1:\(port)" || field == "n*:\(port)" {
                owners.insert(pid)
            }
        }
        guard !owners.isEmpty else { return .unknown }
        let allOwned = owners.allSatisfy { $0 == childPID || (ownsGroup && getpgid($0) == childPID) }
        return allOwned ? .server : .other
    }

    private func check(_ child: Process, url: URL, deadline: Date) {
        guard child === process, child.isRunning, case .starting = state else { return }
        guard Date() < deadline else {
            stop(reason: "The server did not become ready in four minutes. Open the server log for details.")
            return
        }
        guard healthCheck == nil, let port = url.port else { return }
        var request = URLRequest(url: url.appendingPathComponent("status"))
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.timeoutInterval = 2
        let childPID = child.processIdentifier
        let group = ownsGroup
        let task = session.dataTask(with: request) { [weak self] data, response, _ in
            var owner: ListenerOwner?
            if let http = response as? HTTPURLResponse, http.url == request.url,
               http.statusCode == 200, let data,
               let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
               json["status"] as? String == "ok", json["server_port"] as? Int == port {
                owner = Self.listenerOwner(port: port, childPID: childPID, ownsGroup: group)
            }
            DispatchQueue.main.async {
                guard let self, child === self.process, case .starting = self.state else { return }
                self.healthCheck = nil
                guard child.isRunning, let owner else { return }
                switch owner {
                case .server:
                    self.cancelMonitoring()
                    self.state = .running(url)
                case .other:
                    self.stop(reason: "Another service took port \(port) during startup. Start the server again to use the next available port.")
                case .unknown:
                    self.stop(reason: "Could not identify the service responding on port \(port). Start the server again.")
                }
            }
        }
        healthCheck = task
        task.resume()
    }

    private func cancelMonitoring() {
        monitor?.invalidate()
        monitor = nil
        healthCheck?.cancel()
        healthCheck = nil
    }

    private func signal(_ child: Process, _ number: Int32) {
        let pid = child.processIdentifier
        guard pid > 0, child.isRunning else { return }
        if ownsGroup, getpgid(pid) == pid {
            kill(-pid, number)
        } else {
            kill(pid, number)
        }
    }

    func stop() { stop(reason: nil) }

    private func stop(reason: String?) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard let child = process else { return }
        if case .stopping = state { return }
        cancelMonitoring()
        stopReason = reason
        state = .stopping
        signal(child, SIGTERM)
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
            guard let self, self.process === child else { return }
            self.signal(child, SIGKILL)
        }
    }

    private func finished(_ child: Process) {
        guard child === process else { return }
        cancelMonitoring()
        try? log?.close()
        log = nil
        process = nil
        ownsGroup = false
        if case .stopping = state {
            let reason = stopReason
            stopReason = nil
            state = reason.map(State.failed) ?? .stopped
        } else {
            state = .failed("The server exited (code \(child.terminationStatus)). Open the server log for details.")
        }
    }
}
