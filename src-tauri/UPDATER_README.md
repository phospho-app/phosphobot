# Tauri Updater Setup

This document explains how to set up and use the Tauri updater for automatic application updates.

## Overview

The updater is configured to:
- Use built-in Tauri dialogs (no frontend code required)
- Check for updates from GitHub releases
- Automatically verify update signatures for security
- Handle the entire update process transparently

## Initial Setup

### 1. Generate Updater Keys

Run the following command to generate your signing keys:

```bash
npm run updater:generate-keys
```

This will:
- Generate a public/private key pair
- Save keys to `updater-keys/` directory
- Update `tauri.conf.json` with the public key
- Provide setup instructions

### 2. Environment Variables

For local development, copy the `.env.example` file to `.env` and update the paths:

```bash
cp .env.example .env
# Then edit .env to set the correct paths
```

Example `.env` file content:
```bash
# Path to private key file (for local development)
TAURI_PRIVATE_KEY=./updater-keys/phosphobot.key

# Password for private key (if you set one)
TAURI_KEY_PASSWORD=
```

### 3. GitHub Actions Setup

The GitHub Actions workflow is already configured with the correct environment variable names. Add these secrets to your GitHub repository:

1. `TAURI_PRIVATE_KEY` - The **content** of your private key file (not the path)
2. `TAURI_KEY_PASSWORD` - The password if you set one during key generation

**Important**: The workflow uses the content of the private key file, not the file path. Copy the entire contents of your `updater-keys/phosphobot.key` file into the `TAURI_PRIVATE_KEY` secret.

## Configuration

The updater is configured in `tauri.conf.json`:

```json
{
  "plugins": {
    "updater": {
      "active": true,
      "endpoints": [
        "https://api.github.com/repos/phospho-app/phosphobot/releases/latest"
      ],
      "dialog": true,
      "pubkey": "your-public-key-here"
    }
  }
}
```

### Configuration Options

- `active`: Enable/disable the updater
- `endpoints`: Array of URLs to check for updates
- `dialog`: Use built-in dialogs (set to `true` for automatic handling)
- `pubkey`: Public key for signature verification

## How It Works

### For End Users

1. App automatically checks for updates on startup
2. If an update is available, a dialog appears asking if they want to update
3. If they accept, the update downloads and installs automatically
4. User is prompted to restart the application

### For Developers

1. When you build with `tauri build`, artifacts are automatically signed
2. Publish a GitHub release with the signed artifacts
3. The updater checks the GitHub API for new releases
4. Updates are verified using the public key before installation

## Update Process

### 1. Create a Release

```bash
# Build the application (artifacts will be signed automatically)
npm run build

# The build process will generate signed artifacts in src-tauri/target/release/bundle/
```

### 2. GitHub Release

1. Create a new release on GitHub
2. Upload the signed artifacts from the build
3. The updater will automatically detect the new release

### 3. Update Endpoints

The updater checks these endpoints for updates:
- `https://api.github.com/repos/phospho-app/phosphobot/releases/latest`

The response should contain:
- `url`: Download URL for the update
- `version`: Version number
- `signature`: Signature for verification
- `notes`: Release notes (optional)

## Security

- All updates are cryptographically signed using Ed25519
- The public key is embedded in the application
- Updates are verified before installation
- Invalid signatures are rejected

## Troubleshooting

### Key Generation Issues

If key generation fails, ensure:
- Tauri CLI is installed: `npm install -g @tauri-apps/cli`
- You have write permissions to the project directory

### Build Issues

If signing fails during build:
- Check that `TAURI_PRIVATE_KEY` environment variable is set
- Verify the private key file exists and is readable
- Ensure `TAURI_KEY_PASSWORD` is set if your key has a password

### Update Issues

If updates don't work:
- Check that the public key in `tauri.conf.json` matches your private key
- Verify the GitHub release contains properly signed artifacts
- Check the update endpoint is accessible

## File Structure

```
src-tauri/
├── updater-keys/              # Generated keys (not committed to git)
│   ├── phosphobot.key         # Private key (keep secure!)
│   └── phosphobot.pub         # Public key
├── scripts/
│   └── generate-updater-keys.mjs  # Key generation script
├── tauri.conf.json            # Updater configuration
└── .gitignore                 # Excludes updater-keys/
```

## Important Notes

⚠️ **Security Warning**: 
- Never commit your private key to version control
- Keep your private key secure and backed up
- If you lose your private key, you cannot publish updates to existing users

✅ **Best Practices**:
- Store private keys securely (e.g., in a password manager)
- Use different keys for different applications
- Regularly backup your keys
- Test updates in a staging environment first 