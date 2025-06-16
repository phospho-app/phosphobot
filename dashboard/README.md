# React + TypeScript + Vite

## Install Node and npm and uv

First install nvm. That is a library to manage Node version

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
```

Restart your terminal. You can verify the installation of nvm by runnning:

```bash
command -v nvm
```

The output should be `nvm`
Then you can install latest version of node by running this command:

```bash
nvm install node # "node" is an alias for the latest version
```

## Install uv

We run uv as a python package manager. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

go to teleop folder in pin python 3.10:

```bash
cd teleop
uv python pin 3.10
```
