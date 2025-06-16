# phosphobot

**phosphobot** is a community-driven platform that enables you to train and use VLA (vision language action models) to control real robots.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

- 🕹️ Control your robot with the keyboard, a leader arm, a Meta Quest headset or via API
- 📹 Teleoperate robots to record datasets in LeRobot dataset format
- 🤖 Train action models like ACT, gr00t n1 or Pi0
- 🔥 Use action models to control robots
- 💻 Runs on macOS, Linux and Windows
- 🦾 Compatible with the SO-100, SO-101, WX-250 and AgileX Piper
- 🔧 Extend it with your own robots and cameras

## Getting started

### 1. Get a SO-100 robot

Purchase your Phospho starter pack at [robots.phospho.ai](https://robots.phospho.ai) or build your own robot following the instructions in [the SO-100 repo](https://github.com/TheRobotStudio/SO-ARM100).

### 2. Install the phosphobot server

```bash
# Install it this way
curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash
# Start it this way
phosphobot run
# Upgrade it with brew or with apt
# sudo apt update && sudo apt install phosphobot
# brew update && brew upgrade phosphobot
```

### 3. Make your robot move for the first time!

Go to the webapp at `YOUR_SERVER_ADDRESS:YOUR_SERVER_PORT` (default is `localhost:80`) and click control.

You will be able to control your robot with:

- the keyboard
- a leader arm
- a Meta Quest if you have the phospho teleop app

### 4. Record a dataset

Record a 40 episodes dataset of the task you want the robot to learn.

Check out the [docs](https://docs.phospho.ai/basic-usage/dataset-recording) for more details.

### 5. Train an action model

To train an action model on the dataset you recorded, you can:

- train a model directly from the phosphobot webapp (see [this tutorial](https://docs.phospho.ai/basic-usage/training))
- use your own machine (see [this tutorial](tutorials/00_finetune_gr00t_vla.md) to finetune gr00t n1)

In both cases, you will have a trained model exported to huggingface.

To learn more about training action models for robotics, check out the [docs](https://docs.phospho.ai/basic-usage/training).

### 6. Use the model to control your robot

Now that you have a trained model hosted on huggingface, you can use it to control your robot either:

- directly from the webapp
- from your own code using the phosphobot python package (see [this script](scripts/quickstart_ai_gr00t.py) for an example)

Learn more [in the docs](https://docs.phospho.ai/basic-usage/inference).

Congrats! You just trained and used your first action model on a real robot.

## Examples

The `examples/` directory is the quickest way to see the toolkit in action. Check it out!
Proud of what you build? Share it with the community by opening a PR to add it to the `examples/` directory.

## Advanced Usage

You can directly call the phosphobot server from your own code, using the HTTP API and websocket API.

Go to the interactive docs of the API to use it interactively and learn more about it.
It is available at `YOUR_SERVER_ADDRESS:YOUR_SERVER_PORT/docs`. By default, it is available at `localhost:80/docs`.

We release new versions very often, so make sure to check the API docs for the latest features and changes.

## Supported Robots

We currently support the following robots:

- [SO-100](https://github.com/TheRobotStudio/SO-ARM100)
- [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
- [Koch v1.1](https://github.com/jess-moss/koch-v1-1) (beta)
- WX-250 by Trossen Robotics (beta)
- [AgileX Piper](https://global.agilex.ai/products/piper) (Linux-only, beta)
- [Unitree Go2 Air, Pro, Edu](https://shop.unitree.com/en-fr/products/unitree-go2) (beta)
- [LeCabot](https://github.com/phospho-app/lecabot) (beta)

See this [README](phosphobot/README.md) for more details on how to add support for a new robots or open an issue.

## Join the Community

Connect with other developers and share your experience in our [Discord community](https://discord.gg/cbkggY6NSK)

## Install from source

First, download and install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

Then, to build the frontend and backend, run:

```bash
git clone https://github.com/phospho-app/phosphobot.git
cd phosphobot
make
```

Go to `localhost:80` in your browser to see the dashboard or get the server infos with:

```bash
curl -X 'GET' 'http://localhost/status' -H 'accept: application/json'
```

Some features such as connection to the phospho cloud are not available when installing from source.

## Contributing

We welcome contributions! Some of the ways you can contribute:

- Add support for new AI models
- Add support for new teleoperation controllers
- Add support for new robots and sensors
- Add something you built to the examples
- Improve the dataset collection and manipulation
- Improve the [documentation and tutorials](https://github.com/phospho-app/docs)
- Improve code quality and refacto
- Improve the performance of the app
- Fix issues you faced

## Support

- **Documentation**: Read the [documentation](https://docs.phospho.ai)
- **Community Support**: Join our [Discord server](https://discord.gg/cbkggY6NSK)
- **Issues**: Submit problems or suggestions through [GitHub Issues](https://github.com/phospho-app/phosphobot/issues)

## License

MIT License

---

Made with 💚 by the Phospho community
