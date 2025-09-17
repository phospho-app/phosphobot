import os
import wandb
import modal
import asyncio
import sentry_sdk
from pathlib import Path
from loguru import logger
from subprocess import run
from typing import Optional
from huggingface_hub import HfApi
from phosphobot.am.pi05 import Pi05SpawnConfig
from phosphobot.am.base import TrainingParamsPi05, generate_readme

if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent / "phosphobot" / "phosphobot"
)
pi0_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .env({"GIT_LFS_SKIP_SMUDGE": "1"})
    .uv_pip_install(
        "sentry-sdk",
        "loguru",
        "supabase",
        "huggingface_hub[hf_transfer]",
        "hf_xet",
        "httpx",
        "fastparquet",
        "opencv-python-headless",
        "json-numpy",
        "fastapi",
        "wandb",
        "uvicorn",
        "pytest",
    )
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .workdir("/")
    .run_commands(  # clone openpi source code from last commit, we do this to be able to refresh the build when changing the repo
        "git clone https://github.com/phospho-app/openpi.git /openpi-source && cd /openpi-source && git checkout 6ed610ab3d005061f3b517326524de491e9bb91f"
    )
    .run_commands(
        "uv pip install -e /openpi-source",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_python_source("phosphobot")
)
pi0_volume = modal.Volume.from_name("pi0", create_if_missing=True)

# Config constants
MINUTES = 60  # seconds
HOURS = 60 * MINUTES
FUNCTION_IMAGE = pi0_image
FUNCTION_TIMEOUT_TRAINING = 3 * HOURS
FUNCTION_TIMEOUT_INFERENCE = (
    10 * MINUTES
)  # 10 minutes (includes downloading and loading policy model)
FUNCTION_GPU_TRAINING: list[str | modal.gpu._GPUConfig | None] = ["A100-80GB"]
FUNCTION_GPU_INFERENCE: list[str | modal.gpu._GPUConfig | None] = ["A100-40GB", "L40S"]

app = modal.App("pi0.5-server")


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_INFERENCE,
    timeout=FUNCTION_TIMEOUT_INFERENCE,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        # modal.Secret.from_name("supabase"),
    ],
    volumes={"/data": pi0_volume},
)
async def serve(
    model_id: str,
    server_id: int,
    model_specifics: Pi05SpawnConfig,
    checkpoint: int | None = None,
    timeout: int = FUNCTION_TIMEOUT_INFERENCE,
    q: Optional[modal.Queue] = None,
):
    """
    Pi0 inference server function.

    Args:
        model_id: The model identifier from HuggingFace or local path
        server_id: Database server ID for status tracking
        model_specifics: Pi0-specific configuration
        checkpoint: model checkpoint to load
        timeout: Timeout in seconds
        q: Modal queue to pass tunnel info back to caller (since the function is running in a different process)
    """


def _upload_checkpoint(
    checkpoint_dir: Path, model_name: str, hf_token: str, dataset_name: str
):
    """
    Upload the last 5 available checkpoints to the HF model repo.
    Each checkpoint will be uploaded to its own branch named after the checkpoint number.
    The main branch will contain the latest checkpoint.

    Folder will look like this
    folder/
        config.pkl
        wandb_id.txt
        0/
            _CHECKPOINT_METADATA
            assets/
                ...
            params/
                ...
            train_state/
                ...
        100/
            _CHECKPOINT_METADATA
            assets/
                ...
            params/
                ...
            train_state/
                ...

    We want to upload config.pkl, and the assets and params folders from the last 5 checkpoints.
    Each checkpoint goes to branch named after its number (e.g., "100", "1499").
    The latest checkpoint also goes to the main branch.
    """
    api = HfApi(token=hf_token)

    # Ensure the checkpoint directory exists
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

    # Find all checkpoint directories (numbered directories)
    checkpoint_dirs = [
        d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]

    if not checkpoint_dirs:
        raise ValueError(
            f"No checkpoint directories found in {checkpoint_dir}, available: {list(checkpoint_dir.iterdir())}"
        )

    # Get the last 3 checkpoints by sorting numerically and taking the last 3
    sorted_checkpoints = sorted(checkpoint_dirs, key=lambda x: int(x.name))
    last_3_checkpoints = sorted_checkpoints[
        -3:
    ]  # Get last 3 (or fewer if less than 3 exist)
    latest_checkpoint = last_3_checkpoints[-1]  # The very latest checkpoint

    print(f"Found {len(sorted_checkpoints)} total checkpoints")
    print(
        f"Will upload last {len(last_3_checkpoints)} checkpoints: {[cp.name for cp in last_3_checkpoints]}"
    )
    print(
        f"Latest checkpoint {latest_checkpoint.name} will also be uploaded to main branch"
    )

    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=model_name, exist_ok=True, repo_type="model")
        print(f"Repository {model_name} created/verified")
    except Exception as e:
        print(f"Warning: Could not create repository: {e}")

    def upload_checkpoint_to_branch(checkpoint: Path, branch_name: str):
        """Helper function to upload a single checkpoint to a specific branch"""
        checkpoint_name = checkpoint.name
        print(f"Uploading checkpoint {checkpoint_name} to branch '{branch_name}'...")

        # Upload config.pkl to this branch
        config_path = checkpoint_dir / "config.pkl"
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="config.pkl",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            print(f"Uploaded config.pkl to branch {branch_name}")
        else:
            print(f"Warning: config.pkl not found at {config_path}")

        # Upload assets folder from this checkpoint
        norm_json = checkpoint / "assets" / dataset_name
        if norm_json.exists() and norm_json.is_file():
            api.upload_file(
                path_or_fileobj=str(norm_json),
                path_in_repo=f"assets/{dataset_name}",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            print(
                f"Uploaded assets from checkpoint {checkpoint_name} to branch {branch_name}"
            )
        else:
            print(f"Warning: norm json not found at {norm_json}")

        # Upload params folder from this checkpoint
        params_path = checkpoint / "params"
        if params_path.exists() and params_path.is_dir():
            api.upload_folder(
                folder_path=str(params_path),
                path_in_repo="params",
                repo_id=model_name,
                repo_type="model",
                revision=branch_name,
            )
            print(
                f"Uploaded params from checkpoint {checkpoint_name} to branch {branch_name}"
            )
        else:
            print(f"Warning: params folder not found at {params_path}")

    # Upload each of the last 3 checkpoints to their own branches
    for checkpoint in last_3_checkpoints:
        branch_name = checkpoint.name  # Use checkpoint number as branch name

        # Create the branch if it doesn't exist
        try:
            api.create_branch(
                repo_id=model_name, branch=branch_name, repo_type="model", exist_ok=True
            )
            print(f"Branch '{branch_name}' created/verified")
        except Exception as e:
            print(f"Warning: Could not create branch {branch_name}: {e}")

        # Upload checkpoint to its branch
        upload_checkpoint_to_branch(checkpoint, branch_name)

    # Also upload the latest checkpoint to the main branch
    print(f"Uploading latest checkpoint {latest_checkpoint.name} to main branch...")
    upload_checkpoint_to_branch(latest_checkpoint, "main")

    print(
        f"Successfully uploaded {len(last_3_checkpoints)} checkpoints to HuggingFace model: {model_name}"
    )
    print(f"Branches created: {[cp.name for cp in last_3_checkpoints]} + main")
    return f"https://huggingface.co/{model_name}"


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU_TRAINING,
    # 10 extra minutes to make sure the rest of the pipeline is done
    timeout=FUNCTION_TIMEOUT_TRAINING + 10 * MINUTES,
    secrets=[
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"}),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("huggingface"),
    ],
)
async def train(
    training_id: int,
    dataset_name: str,
    wandb_api_key: str | None,
    model_name: str,
    training_params: TrainingParamsPi05,
    user_hf_token: str | None = None,
    private_mode: bool = False,
    timeout_seconds: int = FUNCTION_TIMEOUT_TRAINING,
    **kwargs,
):
    """
    Pi0 training function.

    We run LoRA fine tuning on the provided dataset and upload the trained model to HuggingFace.
    Optionnal: If a wandb_api_key is provided, we log the training to Weights & Biases.
    """
    from datetime import datetime, timezone
    from supabase import Client, create_client

    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    hf_token = user_hf_token or os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError(
            "HF_TOKEN is not available (neither user token nor system token)"
        )

    logger.info(
        f"🚀 Training pi0.5 on {dataset_name} with id {training_id} and uploading to: {model_name}  (private_mode={private_mode})"
    )
    wandb_run_url: str | None = None
    training_params_dict = training_params.model_dump(
        exclude_none=True,
    )
    config_name = training_params_dict.get("config_name", "pi0.5_LoRA_finetune_so100")
    exp_name = training_params_dict.get("exp_name", "pi05_so100_lora_finetune")

    try:
        # Update training status to running
        supabase_client.table("trainings").update(
            {
                "status": "running",
                "requested_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

        wandb_enabled = wandb_api_key is not None

        if wandb_enabled:
            try:
                wandb.login(key=wandb_api_key, verify=True)
            except Exception as e:
                logger.info(
                    f"Failed to login to Weights & Biases: {e}. Disabling Weights & Biases."
                )
                wandb_enabled = False

        # Start by computing normalization stats

        norm_cmd = [
            "uv",
            "run",
            "/openpi-source/scripts/compute_norm_stats.py",
            config_name,
            "--data.repo_id=" + dataset_name,
        ]

        logger.info(f"Training parameters: {training_params_dict}")
        for key, value in training_params_dict.items():
            norm_cmd.append(f"--{key}={value}")

        # Run the command
        logger.info(f"Computing normalization stats with command: {' '.join(norm_cmd)}")

        run(norm_cmd, check=True)
        logger.info("Normalization stats computed successfully")
        # Then run the training

        train_cmd = [
            "uv",
            "run",
            "/openpi-source/scripts/train.py",
            config_name,
            "--data.repo_id=" + dataset_name,
        ]

        # Add any other training parameters that are not None
        logger.info(f"Training parameters: {training_params_dict}")
        for key, value in training_params_dict.items():
            train_cmd.append(f"--{key}={value}")

        logger.info(f"Starting training with command: {' '.join(train_cmd)}")

        output_lines = []

        process = await asyncio.create_subprocess_exec(
            *train_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # 512 KB buffer size, default is 64 KB but is too small for large trainings, will make the training crash
            limit=512 * 1024,
        )

        async def read_output():
            assert process.stdout is not None
            async for line in process.stdout:
                stripped_line = line.decode().strip()
                if wandb_enabled and "wandb: Run" in stripped_line:
                    wandb_run_url = stripped_line.split(" ")[-1]
                    logger.info(f"WandB run URL: {wandb_run_url}")
                logger.debug(stripped_line)
                output_lines.append(stripped_line)

        try:
            await asyncio.wait_for(read_output(), timeout=timeout_seconds)

            # Upload the last available checkpoint to HuggingFace
            try:
                _upload_checkpoint(
                    checkpoint_dir=Path("./checkpoints") / config_name / exp_name,
                    model_name=model_name,
                    hf_token=hf_token,
                    dataset_name=dataset_name,
                )
            except Exception as e:
                logger.error(f"Failed to upload checkpoint: {e}")

            readme = generate_readme(
                model_type="pi0.5",
                dataset_repo_id=dataset_name,
                training_params=training_params,
                return_readme_as_bytes=True,
                wandb_run_url=wandb_run_url,
            )
            api = HfApi(token=hf_token)
            api.upload_file(
                repo_type="model",
                path_or_fileobj=readme,
                path_in_repo="README.md",
                repo_id=model_name,
                token=hf_token,
            )

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"Training process exceeded timeout of {timeout_seconds} seconds. We have uploaded the last checkpoint. Please consider lowering the batch size or number of steps if you wish to train the model longer."
            )

        logger.success(f"Pi0 training {training_id} completed successfully")
        supabase_client.table("trainings").update(
            {
                "status": "succeeded",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", training_id).execute()

    except Exception as e:
        logger.error(f"Pi0 training {training_id} failed: {e}")

        _upload_checkpoint(
            checkpoint_dir=Path("./checkpoints") / config_name / exp_name,
            model_name=model_name,
            hf_token=hf_token,
            dataset_name=dataset_name,
        )

        readme = generate_readme(
            model_type="pi0.5",
            dataset_repo_id=dataset_name,
            training_params=training_params,
            return_readme_as_bytes=True,
            error_traceback=str(e),
            wandb_run_url=wandb_run_url,
        )
        api = HfApi(token=hf_token)
        api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=model_name,
            token=hf_token,
        )

        # Update training status to failed
        try:
            supabase_client.table("trainings").update(
                {
                    "status": "failed",
                    "terminated_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": str(e),
                }
            ).eq("id", training_id).execute()
        except Exception as db_e:
            logger.error(f"Failed to update training status: {db_e}")

        raise e
