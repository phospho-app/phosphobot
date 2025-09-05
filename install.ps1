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
    try {
        Initialize-Environment
        
        Write-Host "Fetching latest release information..." -ForegroundColor Yellow
        
        # Get latest release with error handling
        try {
            $latest_release = Invoke-RestMethod "https://api.github.com/repos/$repo_owner/$repo_name/releases/latest" -ErrorAction Stop
        }
        catch {
            Write-Error "❌ Failed to fetch release information from GitHub"
            Write-Host "   This could be due to:"
            Write-Host "   • No internet connection"
            Write-Host "   • Repository not found or inaccessible"
            Write-Host "   Error details: $($_.Exception.Message)"
            throw "Release fetch failed"
        }
        
        $version = $latest_release.tag_name.TrimStart('v')
        Write-Host "Found latest version: $version" -ForegroundColor Green
        
        # Platform detection
        $arch = if ([System.Environment]::Is64BitOperatingSystem) { "amd64" } else { "i686" }
        $artifact_name = "$app_name-$version-$arch.exe"
        $download_url = $latest_release.assets | Where-Object name -eq $artifact_name | Select-Object -First 1 -ExpandProperty browser_download_url
        
        if (-not $download_url) {
            Write-Error "❌ Could not find Windows binary for version $version"
            Write-Host "   Available assets:"
            $latest_release.assets | ForEach-Object { Write-Host "   • $($_.name)" }
            throw "Binary not found for architecture: $arch"
        }
        
        Write-Host "Binary found: $artifact_name" -ForegroundColor Green
        
        # Create install directory with error handling
        try {
            New-Item -ItemType Directory -Path $install_dir -Force -ErrorAction Stop | Out-Null
            Write-Host "Install directory ready: $install_dir" -ForegroundColor Green
        }
        catch {
            Write-Error "❌ Failed to create install directory: $install_dir"
            Write-Host "   Error: $($_.Exception.Message)"
            throw "Directory creation failed"
        }
        
        # Download with comprehensive error handling
        Write-Host "Downloading $app_name $version..." -ForegroundColor Yellow
        $temp_file = Join-Path $env:TEMP $artifact_name
        
        try {
            # Remove existing temp file if it exists
            if (Test-Path $temp_file) {
                Remove-Item $temp_file -Force
            }
            
            Invoke-WebRequest -Uri $download_url -OutFile $temp_file -ErrorAction Stop
            
            # Verify download completed successfully
            if (-not (Test-Path $temp_file)) {
                throw "Download completed but file not found"
            }
            
            $fileSize = (Get-Item $temp_file).Length
            if ($fileSize -eq 0) {
                throw "Downloaded file is empty (0 bytes)"
            }
            
            Write-Host "✅ Download completed successfully ($([math]::Round($fileSize/1MB, 2)) MB)" -ForegroundColor Green
        }
        catch {
            Write-Error "❌ Download failed!"
            Write-Host "   This could be due to:"
            Write-Host "   • Internet connection interrupted"
            Write-Host "   • Insufficient disk space in temp directory"
            Write-Host "   • Antivirus software blocking the download"
            Write-Host "   • GitHub download servers temporarily unavailable"
            Write-Host "   Error details: $($_.Exception.Message)"
            Write-Host "   Download URL: $download_url"
            
            # Clean up partial download
            if (Test-Path $temp_file) {
                Remove-Item $temp_file -Force -ErrorAction SilentlyContinue
            }
            
            throw "Download failed"
        }
        
        # Install with error handling
        $dest_path = Join-Path $install_dir "$app_name.exe"
        
        try {
            # Backup existing installation if it exists
            if (Test-Path $dest_path) {
                $backup_path = "$dest_path.backup"
                Copy-Item $dest_path $backup_path -Force
                Write-Host "Backed up existing installation" -ForegroundColor Yellow
            }
            
            Move-Item -Path $temp_file -Destination $dest_path -Force -ErrorAction Stop
            Write-Host "✅ Installation completed successfully" -ForegroundColor Green
            
            # Remove backup if installation was successful
            $backup_path = "$dest_path.backup"
            if (Test-Path $backup_path) {
                Remove-Item $backup_path -Force -ErrorAction SilentlyContinue
            }
        }
        catch {
            Write-Error "❌ Failed to install binary"
            Write-Host "   Error: $($_.Exception.Message)"
            
            # Restore backup if it exists
            $backup_path = "$dest_path.backup"
            if (Test-Path $backup_path) {
                try {
                    Move-Item $backup_path $dest_path -Force
                    Write-Host "Restored previous installation from backup" -ForegroundColor Yellow
                }
                catch {
                    Write-Warning "Failed to restore backup: $($_.Exception.Message)"
                }
            }
            
            # Clean up temp file
            if (Test-Path $temp_file) {
                Remove-Item $temp_file -Force -ErrorAction SilentlyContinue
            }
            
            throw "Installation failed"
        }
        
        # Add to PATH if not already present
        Write-Host "Checking PATH..." -ForegroundColor Yellow
        if (-not ($env:Path -split ";" -contains $install_dir)) {
            try {
                Add-Path $install_dir
                Write-Host "✅ Added $install_dir to your PATH" -ForegroundColor Green
            }
            catch {
                Write-Warning "⚠️  Failed to add to PATH automatically: $($_.Exception.Message)"
                Write-Host "You can manually add this directory to your PATH:"
                Write-Host "   $install_dir"
            }
        } else {
            Write-Host "✅ Install directory already in PATH" -ForegroundColor Green
        }
        
        Write-Host "`n🎉 Installation complete! Run with:" -ForegroundColor Green
        Write-Host "    phosphobot run`n" -ForegroundColor Cyan
    }
    catch {
        Write-Host "`n💥 Installation failed!" -ForegroundColor Red
        Write-Host "If the problem persists, you can:"
        Write-Host "• Check your internet connection"
        Write-Host "• Try running the script as administrator"
        Write-Host "• Download manually from: https://github.com/$repo_owner/$repo_name/releases"
        throw
    }
}

function Add-Path($dir) {
    try {
        # Read the *user* PATH
        $userPath = [Environment]::GetEnvironmentVariable('Path','User')
        
        # Bail if it's already there
        if ($userPath -split ';' | Where-Object { $_ -eq $dir }) { 
            return 
        }
        
        # Prepend and write back
        $newUserPath = "$dir;$userPath"
        [Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')
        
        # Update this session so that subsequent calls to e.g. Move‑Item will see the new PATH
        $env:Path = $newUserPath
        
        Write-Host "✅ Added $dir to your USER PATH. You can start using it immediately." -ForegroundColor Green
    }
    catch {
        Write-Error "Failed to modify PATH environment variable"
        throw "PATH modification failed: $($_.Exception.Message)"
    }
}

function Initialize-Environment() {
    Write-Host "Checking environment..." -ForegroundColor Yellow
    
    If ($PSVersionTable.PSVersion.Major -lt 5) {
        Write-Error "❌ PowerShell 5 or later is required"
        Write-Host "   Current version: $($PSVersionTable.PSVersion)"
        Write-Host "   Please upgrade PowerShell to continue"
        throw "PowerShell version too old"
    }
    
    # Ensure execution policy allows scripts
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -notin @('Unrestricted', 'RemoteSigned', 'Bypass')) {
        Write-Error "❌ Execution policy needs to be relaxed"
        Write-Host "   Current policy: $executionPolicy"
        Write-Host "   Run this command to fix: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
        throw "Execution policy too restrictive"
    }
    
    Write-Host "✅ Environment check passed" -ForegroundColor Green
}

# Main execution
if ($Help) {
    Get-Help $PSCommandPath -Detailed
    Exit
}

try {
    Install-Phosphobot
} catch {
    Write-Host "`nFor more help, run: $($MyInvocation.MyCommand.Name) -Help" -ForegroundColor Cyan
    exit 1
}