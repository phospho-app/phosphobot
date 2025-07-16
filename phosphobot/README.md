# phosphobot

**phosphobot** – CLI Toolkit for Robot Teleoperation and Action Models
[![PyPI version](https://img.shields.io/pypi/v/phosphobot?style=flat-square)](https://pypi.org/project/phosphobot/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?style=flat-square)](https://github.com/phospho-app/phosphobot)
[![Discord](https://img.shields.io/discord/1106594252043071509?style=flat-square)](https://discord.gg/cbkggY6NSK)

A simple, community-driven middleware for controlling robots, recording datasets, training action models.

All from your terminal or browser dashboard.

## Features

- **Easy Installation**: python module
- **Web Dashboard**: Instant access to an interactive control panel for teleoperation
- **Dataset Recording**: Record expert demonstrations with a keyboard, in VR, or with a leader arm
- **Model Training & Inference**: Kick off training jobs or serve models through HTTP/WebSocket APIs

## phosphobot installation

### Install the compiled version

Check out [our compiled versions](https://docs.phospho.ai/installation) for installation on MacOS, Windows, and Linux without thinking about Python versions.

### Install the python module

We recommend using [uv](https://github.com/astral-sh/uv):

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add phosphobot to your project
uv add phosphobot
```

You can also use pip.

```bash
pip install phosphobot
```

## Install from source

For development or if you face issues with some submodule or version, you can install phosphobot from source.

1. **Clone the phosphobot repo and fetch submodules.** Make sure you have [git lfs](https://git-lfs.com) installed beforehand

   ```bash
   git clone https://github.com/phospho-app/phosphobot.git
   cd phosphobot
   git submodule update --init --recursive
   ```

2. **Install [uv](https://astral.sh/uv/)** to manage python dependencies. The recommended python version for dev is `3.10`

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then restart your terminal and verify that uv is in your path.

   ```bash
   uv --version # should output: uv 0.7.10
   ```

3. **Install nvm and Node.js.** We recommend to manage Node versions via [nvm](https://github.com/nvm-sh/nvm).

   ```bash
   # Install nvm
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
   ```

   Then restart your terminal and verify:

   ```bash
   command -v nvm   # should output: nvm
   ```

   Finally, install the latest Node.js:

   ```bash
   nvm install node   # “node” is an alias for the latest version
   ```

4. **Build the app.** (Linux and MacOS) From the root of the repo, run:

   ```bash
   make
   ```

   Which is a shortcut for the following command:

   ```
   cd ./dashboard && ((npm i && npm run build && mkdir -p ../phosphobot/resources/dist/ && cp -r ./dist/* ../phosphobot/resources/dist/) || echo "npm command failed, continuing anyway")
   cd phosphobot && uv run phosphobot run --simulation=headless
   ```

   On Windows, run the full command to build the app.

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

## Adding a New Robot

You can extend **phosphobot** by plugging in support for any custom robot. Just follow these steps to install phosphobot from source on any OS:

1. **Install phosphobot from source** (see instructions just above)

2. **Create your robot driver**

   In the directory `phosphobot/phosphobot/hardware` add a new file, e.g. `my_robot.py`. Inside, define a class inheriting from `BaseRobot`:

   ```python
   from phosphobot.hardware.base import BaseRobot

   class MyRobot(BaseRobot):
       def __init__(self, config):
           super().__init__(config)
           # Your initialization here

       ... # Implement the BaseRobot's abstract methods here
   ```

   We use [pybullet](https://pybullet.org/wordpress/) ([docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0)) as a robotics simulation backend. Make sure to add your robot's `urdf` in `phosphobot/resources/urdf`.

3. **Make your robot detectable**
   Open `phosphobot/phosphobot/robot.py` and locate the `RobotConnectionManager` class. Make sure your robot can be detected.

4. **Try in simulation first**

   1. When running phosphobot, use the `--only-simulation` flag and `config.ONLY_SIMULATION` to force the `RobotConnectionManager` to detect your robot, even if it's not connected to hardware. You'll need to change the `RobotConnectionManager` so that it's actually your robot that gets loaded.

   2. When running phosphobot, use the `--simulation=gui` flag to display the pybullet GUI. This way, you can check if keyboard control and VR control actually work in simulation before trying it on hardware. Pay attention to the ways the joints bends and the limits set in the urdf.

   Some general advice: go step by step, don't make any drastic movements, check what values you send to the motors before writing to them, keep your robot near mattresses if ever it falls, keep it away from pets, children, or expensive furniture.

Build and run the app again and ensure your robot gets detected and can be moved. Happy with your changes? Open a pull request! We also recommend you look for testers on [our Discord](https://discord.gg/cbkggY6NSK).

## Start building

- **Docs**: Full user guide at [https://docs.phospho.ai](https://docs.phospho.ai)
- **Discord**: Join us on Discord for support and community chat: [https://discord.gg/cbkggY6NSK](https://discord.gg/cbkggY6NSK)
- **GitHub Repo**: [https://github.com/phospho-app/phosphobot](https://github.com/phospho-app/phosphobot)
- **Examples**: Browse [the examples](https://github.com/phospho-app/phosphobot/tree/main/examples)
- **Contribute**: Open a PR to expand the examples, support more robots, improve the tool

## License

MIT License

Made with 💚 by the Phospho community.
