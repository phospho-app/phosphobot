// Compile with ReleaseInstaller.swift using -swift-version 5 -parse-as-library.
// The URL protocol supplies every response. These tests never contact the publisher.
import CryptoKit
import Foundation

private final class PublisherFixture: URLProtocol {
    static var version = "1.0.0"
    static var payload = Data("first release".utf8)
    static var checksumMatches = true
    static var binaryRequests = 0

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static var formula: String {
        let digest = sha256(checksumMatches ? payload : Data("different data".utf8))
        let base = "https://github.com/phospho-app/homebrew-phosphobot/releases/download/v\(version)"
        return """
        class Phosphobot < Formula
          version "\(version)"
          on_macos do
            if Hardware::CPU.arm?
              url "\(base)/phosphobot-\(version)-arm64.bin"
              sha256 "\(digest)"
            else
              url "\(base)/phosphobot-\(version)-x86_64.bin"
              sha256 "\(digest)"
            end
          end
        end
        """
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let url = request.url!
        let data: Data
        switch url.host {
        case "raw.githubusercontent.com":
            data = Data(Self.formula.utf8)
        case "github.com":
            Self.binaryRequests += 1
            data = Self.payload
        default:
            client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL))
            return
        }
        let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Length": String(data.count)])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

@main
private enum ReleaseInstallerTests {
    static func install(_ installer: ReleaseInstaller) throws -> URL {
        let completed = DispatchSemaphore(value: 0)
        var outcome: Result<URL, Error>?
        installer.installLatest(progress: { _ in }) {
            outcome = $0
            completed.signal()
        }
        precondition(completed.wait(timeout: .now() + 10) == .success, "Timed out waiting for installer")
        return try outcome!.get()
    }

    static func main() throws {
        let fm = FileManager.default
        let temp = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: temp) }
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [PublisherFixture.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let installer = ReleaseInstaller(baseDirectory: temp, session: session)

        for arch in ["arm64", "x86_64"] {
            let release = try ReleaseInstaller.parseFormula(PublisherFixture.formula, architecture: arch)
            precondition(release.asset.lastPathComponent == "phosphobot-1.0.0-\(arch).bin", "Wrong architecture")
        }
        let foreignHost = PublisherFixture.formula.replacingOccurrences(of: "https://github.com", with: "https://example.org")
        precondition((try? ReleaseInstaller.parseFormula(foreignHost, architecture: "arm64")) == nil, "Unexpected publisher accepted")

        let first = try install(installer)
        let firstBytes = try Data(contentsOf: first)
        precondition(firstBytes == PublisherFixture.payload && fm.isExecutableFile(atPath: first.path))
        precondition(installer.installedVersion == "1.0.0" && installer.executableURL == first && !installer.hasPreviousVersion)
        let existing = try install(installer)
        precondition(existing == first && PublisherFixture.binaryRequests == 1, "Downloaded an already installed release")

        PublisherFixture.version = "1.0.1"
        PublisherFixture.payload = Data("second release".utf8)
        PublisherFixture.checksumMatches = false
        precondition((try? install(installer)) == nil, "Accepted an invalid checksum")
        precondition(installer.installedVersion == "1.0.0" && installer.executableURL == first)

        PublisherFixture.checksumMatches = true
        let second = try install(installer)
        let secondBytes = try Data(contentsOf: second)
        precondition(installer.installedVersion == "1.0.1" && secondBytes == PublisherFixture.payload)
        // Reopen from disk so restoring also works when a previous app process performed the upgrade.
        let reopened = ReleaseInstaller(baseDirectory: temp, session: session)
        precondition(reopened.hasPreviousVersion)
        precondition(fm.isExecutableFile(atPath: first.path), "Lost the previous release")
        let restored = try reopened.rollback()
        let restoredBytes = try Data(contentsOf: installer.executableURL)
        precondition(restored && installer.installedVersion == "1.0.0" && firstBytes == restoredBytes && !reopened.hasPreviousVersion)
        let noMoreVersions = try installer.rollback()
        precondition(!noMoreVersions)
        print("PASS: architecture, trusted release URL, install, reuse, checksum, update and rollback")
    }
}
