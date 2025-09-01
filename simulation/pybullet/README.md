# Pybullet simulation server

Run the pybullet simulation server with [uv](<(https://github.com/astral-sh/uv)>). It's only used when running simulation in GUI mode. It uses `python=3.8`. Older versions of Python have bugs where you can't click on the Pybullet window.

## How to run ?

This simulation server is run by the teleop server. Pass `--simulation=gui` when running the server to show the GUI.

```bash
cd ./phosphobot
uv run phosphobot run --simulation=gui --port=8080 --host=127.0.0.1
```

## Run standalone

This creates a new window with a 3d environment that simulates the robot.

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Navigate to the simulation/pybullet folder. Pin python version to 3.8 (this is the only version compatible with pybullet)

```bash
cd ./simulation/pybullet
uv python pin 3.8
```

3. Run the simulation server.

```bash
cd ..
make sim
```

4. In a new terminal, you can now run the main `phosphobot` server, which handles the controller logic.
