# phosphobot

**phosphobot** is a community-driven platform for robotics enthusiasts to share and explore creative projects built with the phospho starter pack.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

This repository contains demo code and community projects developed using the phospho starter pack. Whether you're a beginner or an experienced developer, you can explore existing projects or contribute your own creations.

## Getting started

1. **Get Your Dev Kit**: Purchase your Phospho starter pack at [robots.phospho.ai](https://robots.phospho.ai). Unbox it and set it up following the instructions in the box.

2. **Install the phosphobot server** and run it:

```bash
# Install it this way
curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash
# Start it this way
phosphobot run
# Upgrade it with brew or with apt
# sudo apt update && sudo apt install phosphobot
# brew update && brew upgrade phosphobot
```

3. Use the **phosphobot python client** to interact with the phosphobot server API.

```
pip install --upgrade phosphobot
```

We release new versions very often.

## How to train ACT with LeRobot?

1. **Record a Dataset with phosphobot**: Record a dataset using the app. Do the same gesture 30-50 times (depending on the task complexity) to create a dataset. [Learn more](https://docs.phospho.ai/basic-usage/dataset-recording)

2. **Install LeRobot**. [LeRobot](https://github.com/huggingface/lerobot) by HuggingFace is a research-oriented library for AI training which is still a work in progress. We made a few workarounds to make sure it works reliably. On MacOS, here is a step by step guide.

2.1. Install [uv](https://docs.astral.sh/uv/), a Python environment manager.

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2.2 Create a new directory and install requirements.

```bash
mkdir my_model
cd my_model
uv init
uv add phosphobot git+https://github.com/phospho-app/lerobot
git clone https://github.com/phospho-app/lerobot
```

2.3 On MacOS M1, you need to set this variable for torchcodec to work.

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib:$DYLD_LIBRARY_PATH"
```

2.4 Run the **LeRobot** training script. For example, on Mac M1:

```bash
uv run lerobot/lerobot/scripts/train.py \
 --dataset.repo_id=PLB/simple-lego-pickup-mono-2 \
 --policy.type=act \
 --output_dir=outputs/train/phoshobot_test \
 --job_name=phosphobot_test \
 --policy.device=mps
```

Change the dataset.repo_id to the id of your dataset on Hugging Face.

Change the `--policy.device` flag based on your hardware: `cuda` if you have an NVIDIA GPU, `mps` if you use a MacBook Pro Sillicon, and `cpu` otherwise.

3. **Use the ACT model to control your robot**:

3.1 Launch the ACT server to run inference. This should be running on a beefy GPU machine. Check out our folder [/inference] for more details.

```bash
curl -o server.py https://raw.githubusercontent.com/phospho-app/phosphobot/refs/heads/main/inference/ACT/server.py
```

```bash
uv run server.py --model_id LegrandFrederic/Orange-brick-in-black-box # Replace with <YOUR_HF_MODEL_ID>
```

3.2 Make sure the [phosphobot server](https://docs.phospho.ai/installation) is running to control your robot:

```bash
# Install it this way
curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash
# Start it this way
phosphobot run
```

3.3 Create a script called `my_model/client.py` and copy paste the content below.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "phosphobot",
# ]
# ///
from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import ACT

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

# Instantiate the model
model = ACT()

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
    ]

    # Get the robot state
    state = client.control.read_joints()

    inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}

    # Go through the model
    actions = model(inputs)

    for action in actions:
        # Send the new joint postion to the robot
        client.control.write_joints(angles=action.tolist())
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
```

3.4 Run this script to control your robot using the model:

```
uv run client.py
```

For the full detailed instructions and other model (Pi0, OpenVLA,...), refer to the [docs](https://docs.phospho.ai/basic-usage/inference).

## Join the Community

Connect with other developers and share your experience in our [Discord community](https://discord.gg/cbkggY6NSK)

## Community Projects

Explore projects created by our community members in the [code_examples](./code_examples) directory. Each project includes its own documentation and setup instructions.

## Support

- **Documentation**: Read the [documentation](https://docs.phospho.ai)
- **Community Support**: Join our [Discord server](https://discord.gg/cbkggY6NSK)
- **Issues**: Submit problems or suggestions through [GitHub Issues](https://github.com/phospho-app/phosphobot/issues)

## License

MIT License

---

Made with 💚 by the Phospho community
