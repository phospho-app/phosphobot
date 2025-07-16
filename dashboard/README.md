# React + TypeScript + Vite

This folder contains the dashboard for phosphobot. It's a Typescript React app bundled with Vite. Once transpiled to html and css, the files are copied to `../phosphobot/resources/dist` and are served by the phosphobot server.

## How to setup?

1. Install nvm. That is a library to manage Node version

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
```

2. Restart your terminal. You can verify the installation of nvm by runnning:

```bash
command -v nvm
```

The output should be `nvm`

3. Install latest version of node.

```bash
nvm install node # "node" is an alias for the latest version
```

Building the app requires `node>=20`.

4. To actually launch the phosphobot app, look at "building from source" [in this guide.](../phosphobot/README.md) You still need a few tools to launch the Python server.
