import AppKit

@main
private enum LauncherMain {
    static func main() {
        let app = NSApplication.shared
        let delegate = Launcher()
        app.setActivationPolicy(.regular)
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}

private final class Launcher: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let installer = ReleaseInstaller()
    private let appWindow = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 545, height: 302),
                                     styleMask: [.titled, .closable, .miniaturizable],
                                     backing: .buffered, defer: false)
    private let status = NSTextField(wrappingLabelWithString: "")
    private let versionLabel = NSTextField(labelWithString: "")
    private let endpoint = NSTextField(labelWithString: "")
    private let indicator = NSProgressIndicator()
    private lazy var primary = button("Start server", #selector(toggleServer))
    private lazy var update = button("Check for updates…", #selector(checkForUpdates))
    private lazy var dashboard = button("Open dashboard", #selector(openDashboard))
    private lazy var api = button("API docs", #selector(openAPI))
    private lazy var logs = button("Server log", #selector(openLog))
    private lazy var restore = button("Restore previous version", #selector(restoreVersion))
    private let simulation = NSButton(checkboxWithTitle: "Simulation only (no hardware)",
                                      target: nil, action: nil)

    private var installing = false
    private var installMessage = ""
    private var pendingQuit = false
    private var notice: String?
    private var address: URL?
    private let logURL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/phosphobot-launcher/server.log")
    private lazy var server = BackendServer(logURL: logURL) { [weak self] _ in
        guard let self else { return }
        self.render()
        self.finishQuitIfReady()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let appMenu = NSMenu()
        let menuItem = NSMenuItem()
        menuItem.submenu = appMenu
        menu.addItem(menuItem)
        appMenu.addItem(withTitle: "About phosphobot", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Hide phosphobot", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(withTitle: "Quit phosphobot", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        NSApp.mainMenu = menu

        appWindow.title = "phosphobot"
        appWindow.delegate = self
        appWindow.isReleasedWhenClosed = false
        let heading = NSTextField(labelWithString: "phosphobot server")
        heading.font = .systemFont(ofSize: 23, weight: .semibold)
        versionLabel.textColor = .secondaryLabelColor
        versionLabel.font = .systemFont(ofSize: 12)
        status.font = .systemFont(ofSize: 14)
        status.isSelectable = true
        status.maximumNumberOfLines = 3
        endpoint.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        endpoint.textColor = .secondaryLabelColor
        endpoint.isSelectable = true

        simulation.state = UserDefaults.standard.bool(forKey: "simulationOnly") ? .on : .off
        simulation.setAccessibilityIdentifier("simulationOnly")
        primary.setAccessibilityIdentifier("startStop")
        update.setAccessibilityIdentifier("update")
        indicator.style = .spinning
        indicator.controlSize = .small
        indicator.isDisplayedWhenStopped = false

        let titleLine = row([heading, versionLabel])
        titleLine.alignment = .lastBaseline
        let statusLine = row([indicator, status])
        status.widthAnchor.constraint(lessThanOrEqualToConstant: 448).isActive = true
        let stack = NSStackView(views: [
            titleLine, statusLine, endpoint, simulation, row([primary, update]),
            row([dashboard, api, logs]), restore
        ])
        stack.orientation = .vertical
        stack.spacing = 10
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false
        let content = NSView()
        content.addSubview(stack)
        appWindow.contentView = content
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 22),
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(lessThanOrEqualTo: content.trailingAnchor, constant: -20),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -12)
        ])
        appWindow.center()
        appWindow.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        render()
    }

    private func button(_ title: String, _ action: Selector) -> NSButton {
        let button = NSButton(title: title, target: self, action: action)
        button.bezelStyle = .rounded
        return button
    }

    private func row(_ views: [NSView]) -> NSStackView {
        let stack = NSStackView(views: views)
        stack.orientation = .horizontal
        stack.spacing = 8
        stack.alignment = .centerY
        return stack
    }

    private func render() {
        let version = installer.installedVersion
        versionLabel.stringValue = version.map { "v" + $0 } ?? ""
        address = nil
        var idle = false
        var busy = installing
        var summary = ""
        if installing {
            summary = installMessage
            primary.title = "Please wait…"
        } else {
            switch server.state {
            case .stopped:
                idle = true
                summary = version == nil ? "Install the official server to get started." : "Server stopped."
            case .starting:
                busy = true
                summary = "Starting the server…"
                primary.title = "Stop server"
            case .running(let url):
                address = url
                summary = "Server running."
                primary.title = "Stop server"
            case .stopping:
                busy = true
                summary = "Stopping the server…"
                primary.title = "Stopping…"
            case .failed(let reason):
                idle = true
                summary = reason
            }
            if idle {
                primary.title = version == nil ? "Install & start" : "Start server"
                summary = notice ?? summary
            }
        }
        status.stringValue = summary
        endpoint.stringValue = address?.absoluteString ?? ""
        endpoint.isHidden = address == nil
        busy ? indicator.startAnimation(nil) : indicator.stopAnimation(nil)
        indicator.isHidden = !busy
        var canToggle = !installing
        if case .stopping = server.state { canToggle = false }
        primary.isEnabled = canToggle
        primary.keyEquivalent = idle ? "\r" : ""
        update.isEnabled = idle && version != nil
        update.toolTip = idle ? nil : "Stop the server before updating."
        simulation.isEnabled = idle
        dashboard.isEnabled = address != nil
        api.isEnabled = address != nil
        logs.isEnabled = FileManager.default.fileExists(atPath: logURL.path)
        restore.isHidden = !idle || !installer.hasPreviousVersion
    }

    @objc private func toggleServer() {
        notice = nil
        switch server.state {
        case .starting, .running: server.stop()
        case .stopping: break
        case .stopped, .failed:
            if installer.installedVersion == nil { install(startAfterwards: true) }
            else { startInstalledServer() }
        }
    }

    private func startInstalledServer() {
        let simulationOnly = simulation.state == .on
        UserDefaults.standard.set(simulationOnly, forKey: "simulationOnly")
        server.start(executable: installer.executableURL, simulationOnly: simulationOnly)
    }

    @objc private func checkForUpdates() { install(startAfterwards: false) }

    private func install(startAfterwards: Bool) {
        guard !installing else { return }
        let oldVersion = installer.installedVersion
        notice = nil
        installing = true
        installMessage = "Checking the official release…"
        render()
        installer.installLatest(progress: { [weak self] message in
            DispatchQueue.main.async {
                guard let self, self.installing else { return }
                self.installMessage = message
                self.render()
            }
        }, completion: { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                self.installing = false
                switch result {
                case .success:
                    let changed = oldVersion != self.installer.installedVersion
                    self.notice = changed ? "Installation complete." : "You have the latest server."
                    if startAfterwards && !self.pendingQuit {
                        self.notice = nil
                        self.startInstalledServer()
                    }
                case .failure(let error): self.notice = error.localizedDescription
                }
                self.render()
                self.finishQuitIfReady()
            }
        })
    }

    @objc private func restoreVersion() {
        guard !installing else { return }
        installing = true
        installMessage = "Verifying the previous release…"
        render()
        DispatchQueue.global(qos: .userInitiated).async { [installer, weak self] in
            let result = Result { try installer.rollback() }
            DispatchQueue.main.async {
                guard let self else { return }
                self.installing = false
                switch result {
                case .success(let restored):
                    self.notice = restored ? "Previous version restored." : "No previous version is available."
                case .failure(let error):
                    self.notice = error.localizedDescription
                }
                self.render()
            }
        }
    }

    @objc private func openDashboard() {
        if let address { NSWorkspace.shared.open(address) }
    }

    @objc private func openAPI() {
        if let address { NSWorkspace.shared.open(address.appendingPathComponent("docs")) }
    }

    @objc private func openLog() {
        NSWorkspace.shared.open(logURL)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        appWindow.makeKeyAndOrderFront(nil)
        return true
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        NSApp.terminate(nil)
        return false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        // Downloads use temporary files; the installer atomically activates a
        // verified release. Quitting before activation leaves the previous one usable.
        switch server.state {
        case .stopped, .failed:
            return .terminateNow
        case .starting, .running, .stopping:
            pendingQuit = true
            server.stop()
            return .terminateLater
        }
    }

    private func finishQuitIfReady() {
        guard pendingQuit, !installing else { return }
        switch server.state {
        case .stopped, .failed:
            pendingQuit = false
            NSApp.reply(toApplicationShouldTerminate: true)
        case .starting, .running, .stopping: break
        }
    }
}
