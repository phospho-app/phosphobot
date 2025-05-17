#!/bin/bash

# phosphobot teleop server installation script
# --------------------------------------------
# This script will install and configure phosphobot teleop server on supported platforms.
# Supported platforms: Raspberry Pi, Linux, macOS
# Feel free to review the script before running it.
# Support: contact@phospho.ai

# Run this script on your raspberry pi, linux or macOS device using the following command:
# curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | sudo bash

# Platform detection
PLATFORM="unknown"
OS_TYPE="$(uname)"
IS_WSL=0
IS_RPI=0

# Initialize country variable
COUNTRY=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--country)
            if [[ -n "$2" ]]; then
                COUNTRY="$2"
                shift 2
            else
                echo "Error: --country requires a country code."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Usage: $0 [--country country_code]"
            exit 1
            ;;
    esac
done

# Check for root privileges
check_privileges() {
    if [[ "$PLATFORM" != "darwin" ]]; then
        if [[ $EUID -ne 0 ]]; then
            echo "Error: This script must be run with sudo privileges on Linux-based systems"
            echo "Please run: sudo $0"
            exit 1
        fi
    else
        # For macOS, we don't need sudo for brew commands
        if ! command -v brew &> /dev/null; then
            echo "Error: Homebrew is not installed. Please install Homebrew first."
            echo "Visit https://brew.sh for installation instructions."
            exit 1
        fi
    fi
}

detect_platform() {
    case "$OS_TYPE" in
        "Linux")
            if [ -f /proc/device-tree/model ] && grep -qi "raspberry pi" /proc/device-tree/model; then
                PLATFORM="rpi"
                IS_RPI=1
            elif grep -qi microsoft /proc/version; then
                PLATFORM="wsl"
                IS_WSL=1
            else
                PLATFORM="linux"
            fi
            ;;
        "Darwin")
            PLATFORM="darwin"
            ;;
        *)
            echo "Error: Unsupported platform: $OS_TYPE"
            exit 1
            ;;
    esac
    echo "Detected platform: $PLATFORM"
}

get_install_dir() {
    case "$PLATFORM" in
        "darwin")
            echo "/Users/$(whoami)/Library/Application Support/phosphobot"
            ;;
        *)
            echo "/home/phosphobot"
            ;;
    esac
}

configure_led_monitoring() {
    # LED monitor script installation (existing LED script content remains the same)
    sudo bash -c 'cat > /usr/local/bin/led_monitor.py <<EOL
#!/usr/bin/env python3
import time
import subprocess
import os

LED_PATH = "/sys/class/leds/ACT"

def setup_led():
    try:
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("none")
        return True
    except Exception as e:
        print(f"Error setting up LED: {e}")
        return False

def set_led_state(state):
    try:
        with open(f"{LED_PATH}/brightness", "w") as f:
            f.write("0" if state else "1")
    except Exception as e:
        print(f"Error controlling LED: {e}")

def check_network_status():
    try:
        iwconfig = subprocess.check_output(["iwconfig"], stderr=subprocess.STDOUT).decode("utf-8")
        if "ESSID:off/any" in iwconfig:
            return "disconnected"
        elif "Mode:Master" in iwconfig:
            return "hotspot"
        else:
            return "connected"
    except:
        return "error"

def blink_pattern(status):
    if status == "connected":
        set_led_state(True)
        time.sleep(2)
    elif status == "hotspot":
        for _ in range(2):
            set_led_state(True)
            time.sleep(0.2)
            set_led_state(False)
            time.sleep(0.2)
            set_led_state(True)
            time.sleep(0.2)
            set_led_state(False)
            time.sleep(0.2)
    elif status == "disconnected":
        set_led_state(True)
        time.sleep(1)
        set_led_state(False)
        time.sleep(1)
    else:
        for _ in range(2):
            set_led_state(True)
            time.sleep(0.1)
            set_led_state(False)
            time.sleep(0.1)
        time.sleep(1)

def cleanup():
    try:
        with open(f"{LED_PATH}/trigger", "w") as f:
            f.write("mmc0")
    except Exception as e:
        print(f"Error restoring LED trigger: {e}")

def main():
    if not os.geteuid() == 0:
        print("This script must be run with sudo!")
        return

    if not setup_led():
        return

    try:
        while True:
            status = check_network_status()
            blink_pattern(status)
            time.sleep(3)
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
EOL'

    echo "Making LED monitor script executable..."
    sudo chmod +x /usr/local/bin/led_monitor.py

    echo "Creating LED monitor service..."
    sudo bash -c 'cat > /etc/systemd/system/led-monitor.service <<EOL
[Unit]
Description=Network Status LED Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/led_monitor.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOL'

    sudo systemctl enable led-monitor
    sudo systemctl start led-monitor

    echo "Checking LED monitor service status..."
    sudo systemctl status led-monitor --no-pager
}

install_rpi_specific() {
    echo "Installing Raspberry Pi specific components..."

    sudo apt-get update
    
    # Install RPI-specific dependencies
    sudo apt-get install -y libgl1-mesa-glx dnsmasq v4l-utils
    
    # Configure LED monitoring
    configure_led_monitoring
    
    # Configure Bluetooth
    echo "Installing BT connectivity..."
    if [ -n "$COUNTRY" ]; then
        curl -L https://raw.githubusercontent.com/oulianov/Rpi-SetWiFi-viaBluetooth/refs/heads/main/btwifisetInstall.sh | bash -s -- --yes --country "$COUNTRY"
    else
        curl -L https://raw.githubusercontent.com/oulianov/Rpi-SetWiFi-viaBluetooth/refs/heads/main/btwifisetInstall.sh | bash -s -- --yes
    fi
}

install_darwin_specific() {
    echo "Installing macOS specific components..."

    # Check if Homebrew is installed
    if ! command -v brew >/dev/null 2>&1; then
        echo "Error: Homebrew is not installed. Please install it first."
        echo "Visit https://brew.sh for installation instructions."
        exit 1
    fi

    # Update Homebrew before installing phosphobot
    brew update
}

install_linux_specific() {
    echo "Installing Linux specific components..."

    sudo apt-get update
    
    # Install Linux-specific dependencies
    sudo apt-get install -y curl gnupg libgl1-mesa-glx dnsmasq v4l-utils ethtool can-utils 
}

setup_services() {
        echo "Stopping any existing phosphobot services and processes..."
        sudo systemctl stop phosphobot.service 2>/dev/null || true
        sudo systemctl disable phosphobot.service 2>/dev/null || true
        sudo pkill -f "phosphobot run" 2>/dev/null || true
        sudo systemctl stop phosphobot-update 2>/dev/null || true

        echo "Creating systemd service file for phosphobot..."
        sudo bash -c 'cat > /etc/systemd/system/phosphobot.service <<EOL
[Unit]
Description=Phosphobot FastAPI Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/phosphobot run
Restart=always
WorkingDirectory=/root
Environment="PATH=/usr/local/bin:/usr/bin"

[Install]
WantedBy=multi-user.target
EOL'

        # Create and configure update service
        echo "Creating update script and service..."
    sudo bash -c 'cat > /usr/local/bin/phosphobot-update <<EOL
#!/bin/bash

apt update
apt install -y phosphobot
if [ \$? -eq 0 ]; then
    systemctl restart phosphobot
fi
EOL'

        sudo chmod +x /usr/local/bin/phosphobot-update

        sudo bash -c 'cat > /etc/systemd/system/phosphobot-update.service <<EOL
[Unit]
Description=Phosphobot Update Service
After=network.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/phosphobot-update
RemainAfterExit=no
StandardOutput=journal

[Install]
WantedBy=multi-user.target
EOL'

        sudo bash -c 'cat > /etc/systemd/system/phosphobot-update.timer <<EOL
[Unit]
Description=Run Phosphobot update on boot

[Timer]
OnBootSec=2min
Unit=phosphobot-update.service

[Install]
WantedBy=timers.target
EOL'

        # Reload and enable services
        sudo systemctl daemon-reload
        sudo systemctl enable phosphobot
        sudo systemctl enable phosphobot-update.timer
        sudo systemctl start phosphobot
        sudo systemctl start phosphobot-update.timer

        echo "Checking phosphobot service status..."
        sudo systemctl status phosphobot --no-pager

        echo "Checking update timer status..."
        sudo systemctl list-timers phosphobot-update.timer --no-pager
}

# Main installation flow
main() {
    # Detect platform
    detect_platform
    # Check if we are sudo (to install packages)
    check_privileges

    # Get and create platform-specific installation directory
    INSTALL_DIR=$(get_install_dir)
    echo "Creating installation directory: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR" || {
        echo "Error: Failed to change to installation directory"
        exit 1
    }

    # Platform-specific installation
    case "$PLATFORM" in
        "rpi")
            echo "Starting Raspberry Pi installation..."
            install_rpi_specific
            ;;
        "darwin")
            echo "Starting macOS installation..."
            install_darwin_specific
            ;;
        "linux"|"wsl")
            echo "Starting Linux installation..."
            install_linux_specific
            ;;
        *)
            echo "Error: Unsupported platform"
            exit 1
            ;;
    esac

    # Common installation steps
    if [[ "$PLATFORM" != "darwin" ]]; then
        echo "Installing phosphobot..."
        curl -fsSL https://europe-west1-apt.pkg.dev/doc/repo-signing-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/phosphobot-archive-keyring.gpg

        echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/phosphobot-archive-keyring.gpg] https://europe-west1-apt.pkg.dev/projects/portal-385519 phospho-apt main" | sudo tee /etc/apt/sources.list.d/phosphobot.list > /dev/null
        sudo apt update
        sudo apt install -y phosphobot
    else
        echo "Installing phosphobot via Homebrew..."
        brew tap phospho-app/phosphobot
        brew install phosphobot
    fi

    # Setup services
    if [[ "$PLATFORM" == "rpi" ]]; then
        setup_services
    fi

    # Display status information
    echo "IP address of the device:"

    # Commands to get IP address based on platform
    if [[ "$PLATFORM" == "darwin" ]]; then
        ipconfig getifaddr en0
    else
        hostname -I
    fi

    echo "Installation completed for platform: $PLATFORM"

}

# Run main installation
main