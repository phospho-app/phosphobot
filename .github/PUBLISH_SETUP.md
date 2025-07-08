# GitHub Actions Publishing Setup

This guide explains how to set up the cross-platform publishing workflow for Phosphobot Desktop using GitHub Actions.

## 🔐 Required Secrets

### Core Publishing Secrets

1. **GITHUB_TOKEN** (automatically provided by GitHub)
   - Used for creating releases and uploading assets
   - No setup required - automatically available

### macOS Code Signing (Optional but Recommended)

2. **APPLE_CERTIFICATE**
   - Base64 encoded Developer ID Application certificate (.p12 file)
   - Get from Apple Developer Portal
   ```bash
   base64 -i certificate.p12 | pbcopy
   ```

3. **APPLE_CERTIFICATE_PASSWORD**
   - Password for the .p12 certificate file

4. **KEYCHAIN_PASSWORD**
   - Password for the temporary build keychain (choose any strong password)

5. **APPLE_ID**
   - Your Apple ID email for notarization

6. **APPLE_PASSWORD**
   - App-specific password for your Apple ID
   - Generate at [appleid.apple.com](https://appleid.apple.com) → Security → App-Specific Passwords

7. **APPLE_TEAM_ID**
   - Your Apple Developer Team ID (10-character string)
   - Find in Apple Developer Portal → Membership

### Windows Code Signing (Optional)

8. **WINDOWS_CERTIFICATE**
   - Base64 encoded code signing certificate (.p12 file)
   - Purchase from a Certificate Authority (DigiCert, Sectigo, etc.)
   ```bash
   base64 -i windows-cert.p12 | pbcopy
   ```

9. **WINDOWS_CERTIFICATE_PASSWORD**
   - Password for the Windows certificate

### Tauri Auto-Updater (Optional)

10. **TAURI_SIGNING_PRIVATE_KEY**
    - Private key for Tauri's built-in updater
    - Generate with: `tauri signer generate -w ~/.tauri/myapp.key`

11. **TAURI_SIGNING_PRIVATE_KEY_PASSWORD**
    - Password for the Tauri signing key

### Homebrew Distribution (Optional)

12. **HOMEBREW_TAP_TOKEN**
    - GitHub Personal Access Token with repo permissions
    - For publishing to a Homebrew tap repository
    - Only needed if you want automatic Homebrew cask updates

## 🚀 How to Set Up Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with the exact name listed above

## 📋 Workflow Features

### Cross-Platform Builds
- **macOS**: ARM64 (Apple Silicon) and x86_64 (Intel)
- **Windows**: x86_64 with MSI and NSIS installers
- **Linux**: x86_64 with DEB packages and AppImage

### Automatic Features
- ✅ Cross-platform Python package building with PyApp/Box
- ✅ Code signing for all platforms (when certificates provided)
- ✅ Automatic GitHub releases with detailed descriptions
- ✅ Homebrew cask generation (when configured)
- ✅ Dependency caching for faster builds
- ✅ Draft releases (manual publish required)

### Generated Artifacts

**macOS**:
- `Phosphobot_x.x.x_aarch64.dmg` (Apple Silicon)
- `Phosphobot_x.x.x_x64.dmg` (Intel)
- `Phosphobot.app.tar.gz` files

**Windows**:
- `Phosphobot_x.x.x_x64_en-US.msi` (MSI installer)
- `Phosphobot_x.x.x_x64-setup.exe` (NSIS installer)

**Linux**:
- `phosphobot_x.x.x_amd64.deb` (Debian package)
- `phosphobot_x.x.x_amd64.AppImage` (Universal Linux)

## 🔧 Triggering Builds

### Automatic Triggers
- Push to `main` branch
- Push to `release` branch

### Manual Trigger
- Go to **Actions** → **publish_tauri** → **Run workflow**

## 🛠️ Customization

### Repository Settings
Update these values in the workflow file for your repository:

```yaml
# Line 167: Update repository name
repository: your-username/your-repo-name

# Line 190: Update download repo
--repo your-username/your-repo-name

# Line 203: Update download URL
DOWNLOAD_URL="https://github.com/your-username/your-repo-name/releases/download/phosphobot-v${VERSION}/${{ steps.shasum.outputs.file_name }}"

# Line 206: Update verified domain
verified: "github.com/your-username/your-repo-name/"

# Line 210: Update homepage
homepage "https://github.com/your-username/your-repo-name"
```

### Branch Configuration
To change which branches trigger builds:

```yaml
on:
  push:
    branches:
      - main        # Change to your main branch
      - release     # Add/remove branches as needed
```

## 🔍 Troubleshooting

### Common Issues

1. **Code Signing Failures**
   - Verify certificate is valid and not expired
   - Check that Apple ID and Team ID are correct
   - Ensure app-specific password is generated correctly

2. **Build Failures**
   - Check that all dependencies are properly cached
   - Verify Python and Node.js versions are compatible
   - Ensure Rust toolchain has correct targets installed

3. **Release Creation Failures**
   - Check that `GITHUB_TOKEN` has proper permissions
   - Verify repository settings allow Actions to create releases
   - Check that branch protection rules allow automated commits

### Debug Steps

1. **Enable Debug Logging**
   ```yaml
   env:
     ACTIONS_STEP_DEBUG: true
     ACTIONS_RUNNER_DEBUG: true
   ```

2. **Check Build Logs**
   - Go to Actions tab → Failed workflow → Expand failing step
   - Look for specific error messages in the logs

3. **Test Local Build**
   ```bash
   # Test the build process locally
   cd src-tauri
   npm run python:package:build
   npm run build
   ```

## 📚 References

- [Tauri GitHub Actions Documentation](https://v2.tauri.app/distribute/pipelines/github/)
- [tauri-action Repository](https://github.com/tauri-apps/tauri-action)
- [Apple Code Signing Guide](https://v2.tauri.app/distribute/sign/macos/)
- [Windows Code Signing Guide](https://v2.tauri.app/distribute/sign/windows/)
- [Linux Packaging Guide](https://v2.tauri.app/distribute/sign/linux/)

## 🎯 Next Steps

1. Set up the required secrets (at minimum: code signing certificates)
2. Test the workflow with a manual trigger
3. Verify that artifacts are generated correctly
4. Set up repository settings for automatic releases
5. Configure branch protection and review requirements
6. Set up Homebrew tap repository (optional)

The workflow will create draft releases that you can review and publish manually for maximum control over your distribution process. 