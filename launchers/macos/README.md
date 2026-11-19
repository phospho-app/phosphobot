# phosphobot for macOS

The disk image installs a native launcher for the phosphobot robotics server. Use the launcher to install or update the server, start it, and open its dashboard in your browser.

## Install and run

1. Open the universal disk image and drag **phosphobot** into **Applications**. Eject the image and open the installed application.
2. Choose **Install & start** in the launcher window. The first download requires internet access to GitHub. The launcher reads the published Homebrew formula, downloads the matching server binary, and verifies its SHA-256 checksum before activating it.
3. When the server is ready, open its dashboard. The launcher selects an available local port between 8020 and 8039. Use simulation mode to try the backend without a camera or robot. The Stop control closes the server started by the launcher.

Stop the server and choose **Check for updates…** to install a server update. The launcher retains the previous verified server version for rollback. The managed binaries and release state are stored in `~/Library/Application Support/phosphobot-launcher`. The launcher and server are updated separately. To upgrade the launcher itself, quit it and replace the application with a newer disk image.

The launcher targets macOS 13 and later. The server and connected devices may have their own OS, driver, or permission requirements. macOS may block an ad hoc development build received over the internet. For testing, build the source locally. For distribution, publish a Developer ID-signed, notarized disk image.

## Build

Run this command from the repository root on a Mac with Xcode Command Line Tools:

```sh
launchers/macos/build.sh ./dist/macos
```

The output directory contains `phosphobot.app` and `phosphobot-macos-universal.dmg`. Building on either Apple Silicon or Intel produces a universal application with native `arm64` and `x86_64` executables. The existing backend release pipeline continues to supply the server binaries. The build uses the repository's square phospho logo and includes its license in the application bundle.

Development builds use an ad hoc signature. Set `CODESIGN_IDENTITY` to a Developer ID Application identity in the builder's keychain to sign the application and disk image. This option enables the hardened runtime for the application. Submit the signed image for Apple notarization and staple its ticket before public distribution.

## Verify the disk image

After building, run the following from the repository root. This mounts the disk read-only, checks the Applications shortcut and verifies the signature and both executable architectures.

```sh
(
    set -eu
    dmg="dist/macos/phosphobot-macos-universal.dmg"
    mount_dir="$(mktemp -d)"
    hdiutil verify "$dmg"
    hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" "$dmg"
    trap 'hdiutil detach "$mount_dir" >/dev/null && rmdir "$mount_dir"' EXIT
    test "$(readlink "$mount_dir/Applications")" = /Applications
    app="$mount_dir/phosphobot.app"
    codesign --verify --all-architectures --strict --verbose=2 "$app"
    codesign -dvv "$app" 2>&1
    xcrun lipo -verify_arch arm64 "$app/Contents/MacOS/phosphobot-launcher"
    xcrun lipo -verify_arch x86_64 "$app/Contents/MacOS/phosphobot-launcher"
    test -s "$app/Contents/Resources/phosphobot.icns"
)
```

For an ad hoc build, the signature description contains `Signature=adhoc`. A publisher-signed build lists its signing authorities instead.

## Exercise the installer and a real backend

Compile and run the installer tests on macOS. The tests supply release metadata and binary fixtures locally:

```sh
cd launchers/macos
test_dir="$(mktemp -d)"
xcrun swiftc -swift-version 5 -parse-as-library ReleaseInstaller.swift tests/ReleaseInstallerTests.swift -o "$test_dir/installer-tests"
"$test_dir/installer-tests"
```

To exercise server startup and shutdown, use a matching official release executable. The launcher can install the executable, or you can download the binary identified by the [published macOS formula](https://github.com/phospho-app/homebrew-phosphobot/blob/main/Formula/phosphobot.rb) and verify its SHA-256 before running it. From the same directory:

```sh
xcrun swiftc -swift-version 5 -parse-as-library BackendServer.swift tests/BackendSmoke.swift -o "$test_dir/backend-smoke"
"$test_dir/backend-smoke" /absolute/path/to/phosphobot
```

The smoke test launches the server on loopback in simulation mode, fetches the dashboard, Swagger assets and API schema, then stops the server and checks that its port closed.
