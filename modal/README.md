# Modal

We use modal as a GPU provider, phosphobot can train and run inference on several robotics models by running the scripts in this folder through modal.

## Installation

Login to modal and click on `phospho`/`your_own_space` in the webpage that opens in your browser.

```bash
cd modal
make install
```

## Dev setup

Since modal is cloud only, you might run into issues if multiple people want to work in the test environment.

Check the current environments available with:

```bash
uv run modal environment list
```

To avoid this, create your own modal environment that start with `test-`:

```bash
uv run modal environment create YOUR_ENV_NAME
```

Then, you can switch to your environment with:

```bash
uv run modal config set-environment PaliGemma
```

Create the nescessary volumes:

```bash
uv run modal volume create PaliGemma
uv run modal volume create act
uv run modal volume create gr00t-n1
```

Create the necessary secrets:

```bash
uv run modal secret create huggingface HF_TOKEN=YOUR_TOKEN
uv run modal secret create supabase SUPABASE_KEY=YOUR_SUPABASE_KEY SUPABASE_SERVICE_ROLE_KEY=YOUR_SUPABASE_SERVICE_ROLE_KEY SUPABASE_URL=YOUR_SUPABASE_URL
uv run modal secret create stripe STRIPE_API_KEY=YOUR_STRIPE_API_KEY STRIPE_WEBHOOK_SECRET=YOUR_STRIPE_WEBHOOK_SECRET
```

You can now deploy all apps to your environment:

```bash
make deploy_all
```

You are all set!

## Deploy the admin app

```bash
cd modal
make deploy_admin
```

## Deploy the gr00t model

```bash
cd modal
make deploy_gr00t
```

## Structure

- `admin`: FastAPI server used by `phosphobot` to spin up gpus and models
  - `fastapi_app`: a fastapi app to expose endpoints
- `gr00t`: gr00t-n1 model image
  - `spawn`: a modal function to launch the gr00t inference server and serve it over a tunnel
  - `train`: a modal function to train a gr00t model
- `act`: ACT model image
  - `train`: a modal function to train an ACT model
- `paligemma`: paligemma model image
  - `warmup_model`: a modal function to load the model
  - `detect_object`: a modal function to use the model

## Environments

We have 2 environments we use in Modal:

- `test`: the test environment
- `production`: the production environment

Feel free to mess with the `test` env (deploy stuff,...). Apps and functions of the `production` env are deployed by the CI/CD.

> Make sure you deploy from the right environment in the CLI. See the [Modal docs](https://modal.com/docs/guide/environments) for more details.

## Run the server

Trigger the function serve with the following command (replace `PLB/GR00T-N1-lego-pickup-mono-2` with the model you want to use):

```bash
modal run gr00t/app.py::serve --model-id "PLB/GR00T-N1-lego-pickup-mono-2"
```

## Adding a model to the modal volume

To reduce cold start of the serve() function, we can save the model weights in a modal volume.
TODO: add push to modal volume in Replicate pipeline.

The volume is called `gr00t-n1`.

The volume structure is the following:

```
models/
  - hf_username/
    - repo_name/
```

So if you have a huggingface model_id like `hf_username/repo_name`, you can add it to the modal volume with the following command:

```bash
modal volume put gr00t-n1 local/path/to/model models/hf_username/repo_name
```

So if you have the huggingface model_id, you can easily check if we have it in the volume.
