// Native shell for Trainform: starts the bundled server (Python + embedded Postgres) and shows the UI in a web view.
import Cocoa
import WebKit

// Fixed so the web view's origin, and with it the saved login, is the same on every launch.
let port = 47821
let home = URL(string: "http://127.0.0.1:\(port)/")!
let logFile = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Trainform.log")

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKDownloadDelegate {
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
    let web = WKWebView()
    let server = Process()
    let launcherPipe = Pipe()  // the server exits when this closes, even if the app is force-quit
    var quitting = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.mainMenu = mainMenu()
        web.navigationDelegate = self
        web.uiDelegate = self
        web.loadHTMLString("""
            <body style="margin:0;height:100vh;display:grid;place-items:center;background:#05060f;color:#8fa3ff;
            font:15px -apple-system">Starting Trainform…</body>
            """, baseURL: nil)
        window.title = "Trainform"
        window.contentView = web
        window.center()
        window.setFrameAutosaveName("Trainform")
        window.makeKeyAndOrderFront(nil)
        startServer()
    }

    func startServer() {
        let res = Bundle.main.resourceURL!
        server.executableURL = res.appendingPathComponent("python/bin/python3")
        server.arguments = ["-m", "app.desktop", "--port", String(port), "--exit-with-stdin",
                            "--pg-bin", res.appendingPathComponent("postgres/bin").path,
                            "--static-dir", res.appendingPathComponent("frontend").path]
        server.currentDirectoryURL = res.appendingPathComponent("backend")
        // Bytecode is compiled at build time; writing it at runtime would break the bundle's signature.
        server.environment = ProcessInfo.processInfo.environment.merging(["PYTHONDONTWRITEBYTECODE": "1"]) { $1 }
        server.standardInput = launcherPipe
        FileManager.default.createFile(atPath: logFile.path, contents: nil)
        if let log = try? FileHandle(forWritingTo: logFile) {
            server.standardOutput = log
            server.standardError = log
        }
        server.terminationHandler = { _ in
            DispatchQueue.main.async { self.fail("Trainform's server stopped unexpectedly.") }
        }
        do {
            try server.run()
        } catch {
            return fail("Trainform's server could not start: \(error.localizedDescription)")
        }
        waitForServer(until: Date().addingTimeInterval(180))
    }

    func waitForServer(until deadline: Date) {
        URLSession.shared.dataTask(with: home.appendingPathComponent("api/health")) { _, response, _ in
            let up = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async {
                if up {
                    self.web.load(URLRequest(url: home))
                } else if Date() > deadline {
                    self.fail("Trainform's server did not start in time.")
                } else if self.server.isRunning {  // if it exited, terminationHandler reports it
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.waitForServer(until: deadline) }
                }
            }
        }.resume()
    }

    func fail(_ message: String) {
        guard !quitting else { return }
        quitting = true
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = "Details are in \(logFile.path)."
        alert.addButton(withTitle: "Quit")
        alert.addButton(withTitle: "Show Log")
        if alert.runModal() == .alertSecondButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([logFile])
        }
        NSApp.terminate(nil)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        quitting = true
        guard server.isRunning else { return .terminateNow }
        server.terminate()  // SIGTERM: uvicorn shuts down, then app.desktop stops Postgres
        DispatchQueue.main.asyncAfter(deadline: .now() + 30) {
            // Hung: kill it; the next launch replaces the orphaned Postgres.
            if self.server.isRunning { kill(self.server.processIdentifier, SIGKILL) }
        }
        DispatchQueue.global().async {
            self.server.waitUntilExit()
            DispatchQueue.main.async { sender.reply(toApplicationShouldTerminate: true) }
        }
        return .terminateLater
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    // MARK: Browser features the UI uses: confirm(), <input type=file>, target=_blank links, downloads.

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        completionHandler(alert.runModal() == .alertFirstButtonReturn)
    }

    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = parameters.allowsDirectories
        completionHandler(panel.runModal() == .OK ? panel.urls : nil)
    }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url { NSWorkspace.shared.open(url) }  // opens in the default browser
        return nil
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        decisionHandler(navigationAction.shouldPerformDownload ? .download : .allow)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
                  completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = suggestedFilename
        guard panel.runModal() == .OK, let url = panel.url else { return completionHandler(nil) }
        try? FileManager.default.removeItem(at: url)  // the panel already confirmed replacing; WKDownload won't overwrite
        completionHandler(url)
    }

    func mainMenu() -> NSMenu {
        let menus = [
            ("Trainform", [("About Trainform", "orderFrontStandardAboutPanel:", ""), ("Hide Trainform", "hide:", "h"),
                           ("Quit Trainform", "terminate:", "q")]),
            // Without these, copy/paste shortcuts don't reach the web view's text fields.
            ("Edit", [("Undo", "undo:", "z"), ("Redo", "redo:", "Z"), ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                      ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")]),
            ("View", [("Reload", "reload:", "r")]),
            ("Window", [("Minimize", "performMiniaturize:", "m"), ("Close", "performClose:", "w")]),
        ]
        let bar = NSMenu()
        for (title, items) in menus {
            let menu = NSMenu(title: title)
            for (name, action, key) in items {
                menu.addItem(NSMenuItem(title: name, action: Selector(action), keyEquivalent: key))
            }
            bar.addItem(withTitle: title, action: nil, keyEquivalent: "").submenu = menu
        }
        return bar
    }
}

MainActor.assumeIsolated {
    let delegate = AppDelegate()
    NSApplication.shared.delegate = delegate
    NSApplication.shared.run()
}
