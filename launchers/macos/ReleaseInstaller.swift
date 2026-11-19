import CryptoKit
import Darwin
import Foundation

/// Installs the same standalone binary as the publisher's Homebrew formula.
final class ReleaseInstaller {
    struct Release: Codable, Equatable {
        let version: String
        let architecture: String
        let sha256: String

        var filename: String { "phosphobot-\(version)-\(architecture).bin" }
        var asset: URL {
            URL(string: "https://github.com/phospho-app/homebrew-phosphobot/releases/download/v\(version)/\(filename)")!
        }
    }

    private struct State: Codable {
        let current: Release
        let previous: Release?
    }

    private struct InstallError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    private static let formula = URL(string: "https://raw.githubusercontent.com/phospho-app/homebrew-phosphobot/main/Formula/phosphobot.rb")!
    private let fm = FileManager.default
    private let directory: URL
    private let session: URLSession

    init(baseDirectory: URL? = nil, session: URLSession = .shared) {
        self.directory = baseDirectory ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("phosphobot-launcher", isDirectory: true)
        self.session = session
    }

    private var stateFile: URL { directory.appendingPathComponent("release.json") }
    private func state() throws -> State? {
        guard fm.fileExists(atPath: stateFile.path) else { return nil }
        let saved = try JSONDecoder().decode(State.self, from: Data(contentsOf: stateFile))
        for release in [saved.current, saved.previous].compactMap({ $0 }) {
            guard release.sha256.range(of: #"\A[0-9a-f]{64}\z"#, options: .regularExpression) != nil else {
                throw InstallError("The saved release record is invalid.")
            }
        }
        return saved
    }
    private func save(_ newState: State) throws {
        try JSONEncoder().encode(newState).write(to: stateFile, options: .atomic)
    }
    private func binary(for release: Release) -> URL {
        directory.appendingPathComponent("release-" + release.sha256).appendingPathComponent("phosphobot")
    }

    var installedVersion: String? {
        guard let installed = try? state(), fm.isExecutableFile(atPath: binary(for: installed.current).path) else { return nil }
        return installed.current.version
    }
    var executableURL: URL {
        guard let installed = try? state() else { return directory.appendingPathComponent("phosphobot") }
        return binary(for: installed.current)
    }
    var hasPreviousVersion: Bool {
        guard let previous = (try? state())?.previous else { return false }
        return fm.isExecutableFile(atPath: binary(for: previous).path)
    }

    static func parseFormula(_ formula: String, architecture: String) throws -> Release {
        // Parse data only. Never evaluate or invoke Ruby, Homebrew or an installer script.
        func capture(_ pattern: String) -> String? {
            guard let regex = try? NSRegularExpression(pattern: pattern),
                  let match = regex.firstMatch(in: formula, range: NSRange(formula.startIndex..., in: formula)),
                  let range = Range(match.range(at: 1), in: formula) else { return nil }
            return String(formula[range])
        }
        guard let version = capture(#"(?m)^\s*version\s+"([0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?)"\s*$"#),
              ["arm64", "x86_64"].contains(architecture) else {
            throw InstallError("The publisher's macOS release version or architecture could not be read.")
        }
        let filename = "phosphobot-\(version)-\(architecture).bin"
        let url = "https://github.com/phospho-app/homebrew-phosphobot/releases/download/v\(version)/\(filename)"
        let pattern = #"(?m)^[ \t]*url[ \t]+""# + NSRegularExpression.escapedPattern(for: url)
            + #""[ \t]*\r?\n[ \t]*sha256[ \t]+"([A-Fa-f0-9]{64})""#
        guard let checksum = capture(pattern)?.lowercased() else {
            throw InstallError("The macOS \(architecture) download and its SHA-256 checksum are missing from the publisher's formula.")
        }
        return Release(version: version, architecture: architecture, sha256: checksum)
    }

    private func validate(_ response: URLResponse, hosts: Set<String>) throws -> HTTPURLResponse {
        guard let http = response as? HTTPURLResponse,
              http.url?.scheme == "https", hosts.contains(http.url?.host ?? "") else {
            throw InstallError("The download did not come from the publisher's HTTPS release service.")
        }
        guard http.statusCode == 200 else {
            throw InstallError("The publisher returned HTTP \(http.statusCode). Try the update again later.")
        }
        return http
    }

    private func checksum(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 1024 * 1024), !chunk.isEmpty {
            hash.update(data: chunk)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Stage and publish under a file lock so a second launcher cannot corrupt the previous-version pointer.
    private func withLock<T>(_ body: () throws -> T) throws -> T {
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let fd = open(directory.appendingPathComponent("install.lock").path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw InstallError("Cannot open the launcher installation lock.") }
        defer { close(fd) }
        guard flock(fd, LOCK_EX) == 0 else { throw InstallError("Cannot lock the launcher installation.") }
        defer { flock(fd, LOCK_UN) }
        return try body()
    }

    private func activate(_ downloaded: URL, release: Release) throws -> URL {
        try withLock {
            let old = try state()
            let destination = binary(for: release)
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            // The install lock allows a single staging name. Reclaim an incomplete copy left by a quit.
            let temporary = directory.appendingPathComponent(".incoming-phosphobot")
            if fm.fileExists(atPath: temporary.path) { try fm.removeItem(at: temporary) }
            defer { try? fm.removeItem(at: temporary) }
            try fm.copyItem(at: downloaded, to: temporary)
            try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: temporary.path)
            guard rename(temporary.path, destination.path) == 0 else {
                throw InstallError("Could not activate the verified executable on disk.")
            }
            let previous = old?.current == release ? old?.previous : old?.current
            try save(State(current: release, previous: previous))
            if let obsolete = old?.previous, obsolete.sha256 != release.sha256, obsolete.sha256 != previous?.sha256 {
                try? fm.removeItem(at: binary(for: obsolete).deletingLastPathComponent())
            }
            return destination
        }
    }

    /// Progress messages and completion are delivered off the main thread.
    func installLatest(progress: @escaping (String) -> Void, completion: @escaping (Result<URL, Error>) -> Void) {
        Task.detached(priority: .utility) { [self] in
            do {
                progress("Checking the latest phosphobot release…")
                let (data, response) = try await session.data(for: URLRequest(url: Self.formula, cachePolicy: .reloadIgnoringLocalCacheData))
                _ = try validate(response, hosts: ["raw.githubusercontent.com"])
                guard let formula = String(data: data, encoding: .utf8) else { throw InstallError("Could not read the release formula.") }
                #if arch(arm64)
                let architecture = "arm64"
                #else
                let architecture = "x86_64"
                #endif
                let release = try Self.parseFormula(formula, architecture: architecture)
                if let installed = try state(), installed.current == release,
                   fm.isExecutableFile(atPath: binary(for: release).path),
                   try checksum(binary(for: release)) == release.sha256 {
                    completion(.success(binary(for: release)))
                    return
                }
                progress("Downloading phosphobot \(release.version) for \(architecture)…")
                let (file, responseAsset) = try await session.download(from: release.asset)
                defer { try? fm.removeItem(at: file) }
                let http = try validate(responseAsset, hosts: ["github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"])
                let bytes = (try fm.attributesOfItem(atPath: file.path)[.size] as? NSNumber)?.int64Value ?? 0
                guard bytes > 0, http.expectedContentLength < 0 || bytes == http.expectedContentLength else {
                    throw InstallError("The macOS binary download is incomplete.")
                }
                progress("Verifying the publisher’s SHA-256 checksum…")
                guard try checksum(file) == release.sha256 else { throw InstallError("The downloaded executable failed SHA-256 verification.") }
                let location = try activate(file, release: release)
                completion(.success(location))
            } catch {
                completion(.failure(error))
            }
        }
    }

    @discardableResult func rollback() throws -> Bool {
        try withLock {
            guard let installed = try state(), let previous = installed.previous else { return false }
            let previousBinary = binary(for: previous)
            guard fm.isExecutableFile(atPath: previousBinary.path), try checksum(previousBinary) == previous.sha256 else {
                throw InstallError("The previous executable could not be verified.")
            }
            try save(State(current: previous, previous: nil))
            if installed.current.sha256 != previous.sha256 {
                try? fm.removeItem(at: binary(for: installed.current).deletingLastPathComponent())
            }
            return true
        }
    }
}
