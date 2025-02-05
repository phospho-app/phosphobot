# phosphobot

A community-driven platform for robotics enthusiasts to share and explore creative projects built with the phospho Junior Dev Kit.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

This repository contains demo code and community projects developed using the phospho Junior Dev Kit. Whether you're a beginner or an experienced developer, you can explore existing projects or contribute your own creations.

## Getting Started

1. **Get Your Dev Kit**: Purchase your Phospho Junior Dev Kit at [robots.phospho.ai](https://robots.phospho.ai). Unbox it and set it up following the instructions in the box.

2. **Control your Robot**: Donwload the Meta Quest app, connect it to your robot, start teleoperating it.

3. **Record a Dataset**: Record a dataset using the app. Do the same gesture 30 times to create a dataset.

4. **Install the Package**:

```bash
pip install --upgrade phosphobot
```

5. **LeRobot Integration**: Use [Le Robot](https://github.com/huggingface/lerobot) to train a policy on the dataset you just recorded.

Be sure you install lerobot submodule.
Access `lerobot`directory

```bash
cd lerobot
```

6. **Install LeRobot Dependencies**

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e .
```

7. **(Optional) Weight and Biases**

To use weigth and biases for training metrics, run

```bash
wandb login
```

8. **Launch The Training**
   Launch the training script with the following command from the `lerobot` repository (change the device to `cuda` if you have an NVIDIA GPU or `cpu` if you don't have a GPU):

```bash
sudo python lerobot/scripts/train.py \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --policy.type=<act or diffusion or tdmpc or vqbet> \
  --output_dir=outputs/train/phoshobot_test \
  --job_name=phosphobot_test \
  --device=mps \
  --wandb.enable=true
```

## Next steps

- **Test the model**: Run the following code to test the model you just trained (TODO)
- **Join the Community**: Connect with other developers and share your experience in our [Discord community](https://discord.gg/cbkggY6NSK)

## Contributing

We welcome contributions from the community! Here's how you can participate:

1. Fork this repository
2. Create a new branch for your project
3. Add your code and documentation
4. Submit a Pull Request

Please ensure your code is well-documented and includes:

- Clear setup instructions
- Dependencies list
- Basic usage examples
- Any special requirements

## Community Projects

Explore projects created by our community members in the [code_examples](./code_examples) directory. Each project includes its own documentation and setup instructions.

## Support

- **Documentation**: Read the [documentation](https://docs.phospho.ai)
- **Community Support**: Join our [Discord server](https://discord.gg/cbkggY6NSK)
- **Issues**: Submit problems or suggestions through [GitHub Issues](https://github.com/phospho-app/phosphobot/issues)

## License

MIT License

---

Made with ❤️ by the Phospho community
