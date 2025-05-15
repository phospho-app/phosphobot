**phosphobot** – CLI Toolkit for Robot Teleoperation and Action Models
[![PyPI version](https://img.shields.io/pypi/v/phosphobot?style=flat-square)](https://pypi.org/project/phosphobot/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?style=flat-square)](https://github.com/phospho-app/phosphobot)
[![Discord](https://img.shields.io/discord/1106594252043071509?style=flat-square)](https://discord.gg/cbkggY6NSK)

A simple, community-driven middleware for controlling robots, recording datasets, training action models.

All from your terminal or browser dashboard.

---

## Features

- **Easy Installation** via `pip` or the `uv` package manager
- **Web Dashboard**: Instant access to an interactive control panel for teleoperation
- **Dataset Recording**: Record expert demonstrations with a keyboard, in VR, or with a leader arm
- **Model Training & Inference**: Kick off training jobs or serve models through HTTP/WebSocket APIs

---

## Installation

### 1. Using pip

```bash
pip install phosphobot
```

### 2. Using [uv](https://github.com/astral-sh/uv)

If you already use `uv` to manage Python versions and deps:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add phosphobot to your project
uv add phosphobot
```

---

## Quick Start

Once installed, you can start the phosphobot server instantly.

```bash
# Verify installation and view info
phosphobot info

# Start the teleoperation server (default: localhost:80)
phosphobot run

# For custom port, e.g. 8080
phosphobot run --port 8080
```

If you’re managing via uv:

```bash
uv run phosphobot info
uv run phosphobot run
```

---

## Dashboard & Control

After launching the server, open your browser and navigate to:

```
http://<YOUR_SERVER_ADDRESS>:<PORT>/
```

By default, the address is [localhost:80](localhost:80)

Here you can:

- **Teleoperate** your robot via keyboard, leader arm, or Meta Quest
- **Record** demonstration datasets (40 episodes recommended)
- **Train** and **deploy** action models directly from the UI

---

## Start building

- **Docs**: Full user guide at [https://docs.phospho.ai](https://docs.phospho.ai)
- **Discord**: Join us on Discord for support and community chat: [https://discord.gg/cbkggY6NSK](https://discord.gg/cbkggY6NSK)
- **GitHub Repo**: [https://github.com/phospho-app/phosphobot](https://github.com/phospho-app/phosphobot)
- **Examples**: Browse [the examples](https://github.com/phospho-app/phosphobot/tree/main/examples)
- **Contribute**: Open a PR to expand the examples, support more robots, improve the tool

---

## License

MIT License

Made with 💚 by the Phospho community.
