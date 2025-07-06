# Phosphobot Desktop (Tauri App)

A cross-platform desktop application built with [Tauri v2](https://v2.tauri.app/) that provides a native interface for the Phosphobot robotics development kit.

## 🏗️ Architecture Overview

This application follows Tauri's hybrid architecture:

- **Frontend**: React-based dashboard (in `../dashboard/`)
- **Backend**: Rust application with Tauri plugins 
- **Sidecar**: Python application (`phosphobot`) bundled as a native binary
- **Communication**: Frontend ↔ Rust ↔ Python via HTTP API

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React App     │    │   Rust Backend   │    │  Python Sidecar │
│   (Dashboard)   │◄──►│   (Tauri Core)   │◄──►│   (Phosphobot)  │
│                 │    │                  │    │                 │
│  - UI Controls  │    │  - Window Mgmt   │    │  - Robot APIs   │
│  - Status View  │    │  - File System   │    │  - AI Control   │
│  - Settings     │    │  - Auto-Updater  │    │  - Camera Mgmt  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Features

- 🚀 Native performance with web technology UI
- 🔄 Automatic updates with cryptographic signing
- 🐍 Embedded Python application (no Python installation required)
- 📱 Cross-platform: macOS, Windows, Linux
- 🔐 Security-first design with capability-based permissions

## 📋 Prerequisites

### Development Environment

1. **Rust** (latest stable)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Node.js** (LTS version)
   ```bash
   # Using nvm (recommended)
   nvm install --lts
   nvm use --lts
   ```

3. **Python 3.10+** with [uv](https://docs.astral.sh/uv/)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

4. **Platform-specific dependencies**:

   **macOS**:
   ```bash
   xcode-select --install
   ```

   **Linux (Ubuntu/Debian)**:
   ```bash
   sudo apt update
   sudo apt install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
   ```

   **Windows**:
   - Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install [WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Tauri CLI and frontend dependencies
npm install
npm run dashboard:install

# Install Python dependencies and build tools
cd ../phosphobot
uv sync
uv tool install hatch
uv tool install box-packager
cd ../src-tauri
```

### 2. Development Mode

```bash
# Start development server (auto-reload enabled)
npm run dev

# This will:
# 1. Start the React development server (port 5173)
# 2. Build the Python sidecar if needed
# 3. Launch the Tauri app with hot-reload
```

### 3. Production Build

```bash
# Build everything for production
npm run build

# This creates:
# - Signed application bundles in target/release/bundle/
# - Platform-specific installers (DMG, MSI, DEB, AppImage)
```

## 🛠️ Development Workflow

### Project Structure

```
src-tauri/
├── src/                        # Rust source code
│   ├── main.rs                 # Application entry point
│   └── state.rs                # Global application state
├── capabilities/               # Tauri security permissions
│   ├── default.json            # Core app permissions
│   └── http.json               # HTTP client permissions
├── scripts/                    # Build and utility scripts
│   ├── build-python.mjs        # Python sidecar bundling
│   ├── generate-updater-keys.mjs # Update signing keys
│   └── ...
├── binaries/                   # Bundled Python executables
├── icons/                      # Application icons
├── tauri.conf.json            # Tauri configuration
├── Cargo.toml                 # Rust dependencies
└── package.json               # Node.js dependencies and scripts
```

### Available Scripts

```bash
# Development
npm run dev                     # Start dev server with hot-reload
npm run dashboard:dev           # Start frontend only
npm run python:package:dev      # Start Python backend only

# Building
npm run build                   # Full production build
npm run dashboard:build         # Build frontend only
npm run python:package:build    # Build Python sidecar only

# Python Sidecar Management
npm run python:package:check    # Check if Python rebuild needed
npm run python:package:build:if-needed  # Conditional build
npm run python:package:reset    # Reset Python package state

# Utilities
npm run clean                   # Clean all build artifacts
npm run updater:generate-keys   # Generate update signing keys
```

## 🐍 Python Sidecar Integration

The app includes a Python application as a "sidecar" - a separate process that runs alongside the main app.

### How It Works

1. **Bundling**: Python app is packaged into a single executable using [PyApp](https://ofek.dev/pyapp/) via [Box](https://pypi.org/project/box-packager/)
2. **Distribution**: Bundled executable is embedded in the Tauri app
3. **Runtime**: Tauri spawns the Python process and communicates via HTTP
4. **Lifecycle**: Python process is automatically managed (start/stop/restart)

### Build Process

```bash
# The Python build process:
cd ../phosphobot
uv sync                         # Install dependencies
uvx --from box-packager box package  # Bundle into executable
# Creates: target/release/phosphobot (or .exe on Windows)
```

### Cross-Platform Considerations

- **macOS**: Universal binaries supported (Intel + Apple Silicon)
- **Windows**: Handles both x64 and ARM64 architectures
- **Linux**: Uses AppImage for maximum compatibility
- **Dependencies**: All Python dependencies bundled (no system Python required)

## 🔄 Auto-Updater System

### Overview

The app includes automatic updates using Tauri's built-in updater with cryptographic signature verification.

### How Updates Work

1. **Check**: App checks GitHub releases API on startup
2. **Download**: If newer version available, downloads signed bundle
3. **Verify**: Cryptographically verifies signature before installation
4. **Install**: Replaces app files and prompts for restart
5. **Restart**: User can restart to complete update

### Setup Updater

1. **Generate signing keys**:
   ```bash
   npm run updater:generate-keys
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your key paths
   ```

3. **Add to tauri.conf.json** (done automatically):
   ```json
   {
     "plugins": {
       "updater": {
         "active": true,
         "endpoints": ["https://api.github.com/repos/phospho-app/phosphobot/releases/latest"],
         "dialog": true,
         "pubkey": "your-public-key-here"
       }
     }
   }
   ```

### Security Model

- **Ed25519 signatures**: Industry-standard cryptographic signing
- **Public key verification**: Built into the app, cannot be bypassed
- **HTTPS only**: All update downloads use secure transport
- **User consent**: Built-in dialogs require user approval

## 🔐 Code Signing & Distribution

### macOS Code Signing

**Requirements**:
- Apple Developer Account ($99/year)
- Developer ID Application certificate

**Setup**:
1. **Get certificate from Apple Developer Portal**:
   - Log in to [developer.apple.com](https://developer.apple.com)
   - Certificates, Identifiers & Profiles → Certificates
   - Create "Developer ID Application" certificate
   - Download and install in Keychain

2. **Export certificate**:
   ```bash
   # Export from Keychain as .p12 file
   # Set a strong password
   ```

3. **For GitHub Actions, convert to base64**:
   ```bash
   base64 -i certificate.p12 | pbcopy
   ```

**Environment Variables**:
```bash
# For local signing
APPLE_CERTIFICATE_FILE=path/to/cert.p12
APPLE_CERTIFICATE_PASSWORD=your_password

# For GitHub Actions (as secrets)
APPLE_CERTIFICATE=base64_encoded_p12_content
APPLE_CERTIFICATE_PASSWORD=your_password
APPLE_ID=your@apple.id
APPLE_PASSWORD=app_specific_password  # Generate at appleid.apple.com
APPLE_TEAM_ID=YOUR_TEAM_ID
```

### Windows Code Signing

**Requirements**:
- Code signing certificate from trusted CA (DigiCert, Sectigo, etc.)
- Windows SDK (for signtool.exe)

**Setup**:
1. **Purchase certificate** from Certificate Authority
2. **Install certificate** in Windows Certificate Store
3. **For GitHub Actions**:
   ```bash
   # Export as .p12 and convert to base64
   base64 -i windows-cert.p12 | pbcopy
   ```

**Environment Variables**:
```bash
# For GitHub Actions
WINDOWS_CERTIFICATE=base64_encoded_p12_content
WINDOWS_CERTIFICATE_PASSWORD=cert_password
```

### Linux Packaging

Linux builds create:
- **DEB packages**: For Debian/Ubuntu distributions
- **AppImage**: Universal Linux executable
- **No signing required**: Linux distributions handle package signing

## 📦 GitHub Actions CI/CD

### Workflow Overview

The `.github/workflows/publish_tauri.yml` workflow:

1. **Cross-platform builds**: macOS, Windows, Linux
2. **Python bundling**: Packages Python sidecar for each platform
3. **Code signing**: Signs binaries when certificates provided
4. **GitHub releases**: Creates releases with signed artifacts
5. **Homebrew**: Optionally updates Homebrew cask

### Required GitHub Secrets

**Core Secrets**:
```bash
GITHUB_TOKEN                    # Auto-provided by GitHub
```

**macOS Code Signing**:
```bash
APPLE_CERTIFICATE               # Base64 encoded .p12 file
APPLE_CERTIFICATE_PASSWORD      # Certificate password
APPLE_ID                        # Your Apple ID
APPLE_PASSWORD                  # App-specific password
APPLE_TEAM_ID                   # 10-character team ID
```

**Windows Code Signing**:
```bash
WINDOWS_CERTIFICATE             # Base64 encoded .p12 file
WINDOWS_CERTIFICATE_PASSWORD    # Certificate password
```

**Tauri Auto-Updater**:
```bash
TAURI_PRIVATE_KEY              # Content of private key file
TAURI_KEY_PASSWORD             # Key password (if set)
```

**Optional - Homebrew Distribution**:
```bash
HOMEBREW_TAP_TOKEN             # GitHub token for tap repo
```

### Setting Up Secrets

1. Go to GitHub repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add each secret with exact names listed above

### Triggering Builds

**Automatic triggers**:
- Push to `main` branch
- Push to `release` branch
- Push to `tauri` branch

**Manual trigger**:
- GitHub Actions → **publish_tauri** → **Run workflow**

### Build Artifacts

The workflow generates:

**macOS**:
- `Phosphobot_x.x.x_aarch64.dmg` (Apple Silicon)
- `Phosphobot_x.x.x_x64.dmg` (Intel)
- `Phosphobot.app.tar.gz` (updater bundles)

**Windows**:
- `Phosphobot_x.x.x_x64_en-US.msi` (MSI installer)
- `Phosphobot_x.x.x_x64-setup.exe` (NSIS installer)

**Linux**:
- `phosphobot_x.x.x_amd64.deb` (Debian package)
- `phosphobot_x.x.x_amd64.AppImage` (Universal)

## 🔧 Configuration

### Tauri Configuration (`tauri.conf.json`)

Key sections:

```json
{
  "identifier": "ai.phospho.phosphobot",     // Unique app identifier
  "productName": "Phosphobot",               // Display name
  "version": "0.3.52",                       // App version
  
  "build": {
    "beforeDevCommand": "cd ../dashboard && npm run dev",
    "beforeBuildCommand": "cd ../dashboard && npm run build",
    "devUrl": "http://localhost:5173",
    "frontendDist": "../dashboard/dist"
  },
  
  "bundle": {
    "targets": ["app", "dmg", "msi", "nsis", "deb", "appimage"],
    "externalBin": ["binaries/phosphobot"]    // Python sidecar
  },
  
  "plugins": {
    "updater": {
      "active": true,
      "endpoints": ["https://api.github.com/repos/phospho-app/phosphobot/releases/latest"],
      "dialog": true
    }
  }
}
```

### Security Capabilities

Tauri uses a capability-based security model:

**Core permissions** (`capabilities/default.json`):
- Window management
- File system access
- Shell execution (for Python sidecar)
- Auto-updater functions

**HTTP permissions** (`capabilities/http.json`):
- Local API access (127.0.0.1, localhost)
- Required for frontend-backend communication

### Environment Variables

**Development** (`.env`):
```bash
TAURI_PRIVATE_KEY=./updater-keys/phosphobot.key
TAURI_KEY_PASSWORD=
```

**Production** (CI/CD):
```bash
TAURI_PRIVATE_KEY=content_of_private_key_file
TAURI_KEY_PASSWORD=key_password
```

## 🚨 Security Considerations

### Code Signing Importance

1. **User Trust**: Prevents "Unknown Developer" warnings
2. **Malware Protection**: OS-level verification of app integrity
3. **Auto-Updates**: Required for seamless update experience
4. **Distribution**: Required for Mac App Store, recommended for others

### Best Practices

1. **Certificate Security**:
   - Store certificates securely (password managers)
   - Use strong passwords
   - Regular certificate renewal
   - Separate development/production certificates

2. **Key Management**:
   - Never commit private keys to version control
   - Use environment variables or secret management
   - Regular key rotation for update signing
   - Backup keys securely

3. **Update Security**:
   - Always verify signatures before installation
   - Use HTTPS for all update communications
   - Test updates in staging environment
   - Monitor for failed signature verifications

## 🐛 Troubleshooting

### Common Issues

**Build Failures**:
```bash
# Clean everything and rebuild
npm run clean
rm -rf node_modules ../dashboard/node_modules
npm install
npm run dashboard:install
npm run build
```

**Python Sidecar Issues**:
```bash
# Rebuild Python package
npm run python:package:build
# Test the binary
./binaries/phosphobot --help
```

**Code Signing Issues**:
```bash
# Verify certificate is valid
security find-identity -v -p codesigning

# Check certificate expiration
security find-certificate -a -c "Developer ID Application" -p | openssl x509 -text
```

**Update Issues**:
```bash
# Verify public key matches private key
npm run updater:generate-keys --verify
```

### Debug Mode

Enable debug logging:
```bash
# Set environment variable
export RUST_LOG=debug
npm run dev
```

## 📚 Additional Resources

### Official Documentation
- [Tauri v2 Documentation](https://v2.tauri.app/)
- [Tauri GitHub Action](https://github.com/tauri-apps/tauri-action)
- [Tauri Security Guide](https://v2.tauri.app/concept/security/)

### Platform-Specific Guides
- [macOS Distribution](https://v2.tauri.app/distribute/macos-application-bundle/)
- [Windows Distribution](https://v2.tauri.app/distribute/windows-installer/)
- [Linux Distribution](https://v2.tauri.app/distribute/flatpak/)

### Signing Guides
- [macOS Code Signing](https://v2.tauri.app/distribute/sign/macos/)
- [Windows Code Signing](https://v2.tauri.app/distribute/sign/windows/)

### Advanced Topics
- [Sidecar Development](https://v2.tauri.app/develop/sidecar/)
- [Plugin Development](https://v2.tauri.app/develop/plugins/)
- [Custom Protocols](https://v2.tauri.app/develop/calling-rust/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly on all platforms
5. Submit a pull request

### Development Guidelines

- Follow Rust naming conventions
- Use TypeScript for frontend code
- Add tests for new functionality
- Update documentation
- Ensure cross-platform compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

For more specific information about updater setup, see [UPDATER_README.md](./UPDATER_README.md).
For GitHub Actions setup details, see [PUBLISH_SETUP.md](../.github/PUBLISH_SETUP.md). 