# MuJoCo Simulation Server

## Usage

### With phosphobot server

The simulation server is automatically launched when running:

```bash
cd phosphobot
uv run phosphobot run --simulation=gui
```

### Standalone

```bash
cd simulation/mujoco
uv run python main.py
```

## Installation

```bash
pip install mujoco
```

## Robot Models

Robot models are loaded from `phosphobot/resources/urdf/` and automatically converted from URDF to MJCF format when needed. 