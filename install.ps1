<#
.SYNOPSIS
The installer for phosphobot

.DESCRIPTION
This script installs the latest phosphobot binary from GitHub releases
and adds it to your PATH. Updates replace the existing installation.

.PARAMETER Help
Print help
#>

param (
    [Parameter(HelpMessage = "Print Help")]
    [switch]$Help
)

$app_name = "phosphobot"
$repo_owner = "phospho-app"
$repo_name = "homebrew-phosphobot"
$install_dir = Join-Path $env:USERPROFILE ".local/bin"

function Install-Phosphobot {
    Initialize-Environment

    # Get latest release
    $latest_release = Invoke-RestMethod "https://api.github.com/repos/$repo_owner/$repo_name/releases/latest"
    $version = $latest_release.tag_name.TrimStart('v')
    
    # Platform detection
    $arch = if ([System.Environment]::Is64BitOperatingSystem) { "amd64" } else { "i686" }
    $artifact_name = "$app_name-$version-$arch.exe"
    $download_url = $latest_release.assets | Where-Object name -eq $artifact_name | Select-Object -First 1 -ExpandProperty browser_download_url

    if (-not $download_url) {
        throw "Could not find Windows binary for version $version"
    }

    # Create install directory
    New-Item -ItemType Directory -Path $install_dir -Force | Out-Null

    # Download and install
    Write-Host "Downloading $app_name $version..."
    $temp_file = Join-Path $env:TEMP $artifact_name
    Invoke-WebRequest -Uri $download_url -OutFile $temp_file

    # Rename to phosphobot.exe
    $dest_path = Join-Path $install_dir "$app_name.exe"
    Move-Item -Path $temp_file -Destination $dest_path -Force

    # Add to PATH if not already present
    Write-Host "Checking PATH"
    if (-not ($env:Path -split ";" -contains $install_dir)) {
        Add-Path $install_dir
        Write-Host "Added $install_dir to your PATH"
    }

    Write-Host "`nInstallation complete! Run with:"
    Write-Host "    phosphobot run`n"
}

function Add-Path($dir) {
    # Read the *user* PATH
    $userPath = [Environment]::GetEnvironmentVariable('Path','User')
    # Bail if it’s already there
    if ($userPath -split ';' | Where-Object { $_ -eq $dir }) { return }

    # Prepend and write back
    $newUserPath = "$dir;$userPath"
    [Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')

    # Update this session so that subsequent calls to e.g. Move‑Item will see the new PATH
    $env:Path = $newUserPath

    Write-Host "✅ Added `$dir` to your USER PATH. You can start using it immediately."
}


function Initialize-Environment() {
    If ($PSVersionTable.PSVersion.Major -lt 5) {
        throw "PowerShell 5 or later is required"
    }

    # Ensure execution policy allows scripts
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -notin @('Unrestricted', 'RemoteSigned', 'Bypass')) {
        throw "Execution policy needs to be relaxed (run: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser)"
    }
}

# Main execution
if ($Help) {
    Get-Help $PSCommandPath -Detailed
    Exit
}

try {
    Install-Phosphobot
} catch {
    Write-Error $_
    exit 1
}
