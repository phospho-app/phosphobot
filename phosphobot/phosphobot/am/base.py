import random
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional

import av
import numpy as np
import requests  # type: ignore
from huggingface_hub import HfApi
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from phosphobot.models import InfoModel, ModelConfigurationResponse
from phosphobot.utils import get_hf_token

# Disable PyAV logs
av.logging.set_level(None)


class ActionModel(ABC):
    """
    A PyTorch model for generating robot actions from robot state, camera images, and text prompts.
    Inspired by the simplicity and flexibility of pytorch-pretrained-bert.
    """

    def __init__(self, server_url: str = "http://localhost", server_port: int = 8080):
        """
        Initialize the ActionModel.

        Args:
            model_name (str): Name of the pre-trained model (e.g., "PLB/pi0-so100-orangelegobrick-wristcam").
            revision: default None which will resolve to main
        """
        self.server_url = server_url
        self.server_port = server_port

    @classmethod
    def fetch_and_get_configuration(cls, model_id: str) -> ModelConfigurationResponse:
        """
        Fetch the model from Hugging Face and get the configuration.
        Args:
            model_id (str): Model ID on Hugging Face.
        Returns:
            video_keys, list[str]: List of configuration keys.
            checkpoints, list[str]: List of available checkpoints.
        """
        raise NotImplementedError(
            f"This method is not implemented in {cls.__name__}. You need to implement it in your subclass."
        )

    @abstractmethod
    def sample_actions(self, inputs: dict) -> np.ndarray:
        """
        Select a single action.

        Args:
            inputs (dict): Dictionary with keys:
                - "state": Tensor or list of floats representing robot state.
                - "images": List of images (numpy arrays or tensors).
                - "prompt": String text prompt (optional for ACT).

        Returns:
            np.ndarray: Sequence of actions (shape: [max_seq_length, n_actions]).
        """
        raise NotImplementedError("""You cannot directly call the ActionModel class.
                                  You need to use an implementation ( ACT, PI0,...) or implement you own class.""")

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Makes the model instance callable, delegating to the forward method.

        Args:
            *args: Variable positional arguments passed to forward.
            **kwargs: Variable keyword arguments passed to forward.

        Returns:
            The output of the forward method.
        """
        return self.sample_actions(*args, **kwargs)


class TrainingParamsAct(BaseModel):
    """
    Training parameters are left to None by default and are set depending on the dataset in the training pipeline.
    """

    class Config:
        extra = "allow"

    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for training, we run this on an A10G. Leave it to None to auto-detect based on your dataset",
        gt=0,
        le=150,
    )
    steps: Optional[int] = Field(
        default=None,
        description="Number of training steps. Leave it to None to auto-detect based on your dataset",
        gt=0,
        le=1_000_000,
    )
    save_freq: int = Field(
        default=5_000,
        description="Number of steps between saving the model.",
        gt=0,
        le=1_000_000,
    )


DEFAULT_INSTRUCTION = "e.g.: green lego brick, red ball, blue plushy..."


class TrainingParamsActWithBbox(TrainingParamsAct):
    """
    Training parameters for ACT with bounding box
    """

    target_detection_instruction: str = Field(
        default=DEFAULT_INSTRUCTION,
        examples=["red/orange lego brick", "brown plushy", "blue ball"],
        description="Instruction for the target object to detect, e.g. 'red/orange lego brick'",
        min_length=4,
    )
    image_key: str = Field(
        default="main",
        examples=["main", "images.main"],
        description="Key for the image to run detection on, e.g. 'main' or 'images.main'",
        min_length=1,
    )

    # Optional field with the image keys to keep
    image_keys_to_keep: list[str] = Field(
        default_factory=list,
        description="Optional list of image keys to keep. If none, all image keys will be dropped.",
    )

    @field_validator("target_detection_instruction", mode="before")
    def validate_target_detection_instruction(cls, instruction: str) -> str:
        # If the instruction is equal to the default, we raise an error
        if instruction == DEFAULT_INSTRUCTION:
            raise ValueError(
                "Please provide a valid object to detect, e.g. 'red/orange lego brick' or 'brown plushy'."
            )
        elif any(
            word in instruction.lower()
            for word in ["detect", "find", "locate", "pick", "pickup", "grab"]
        ):
            raise ValueError(
                "Only write the object to detect, e.g. 'red/orange lego brick' or 'brown plushy'. Do not include verbs like 'detect', 'find', 'locate', 'pick up', or 'grab'."
            )
        return instruction


class TrainingParamsPi05(BaseModel):
    """
    Training parameters for Pi0 model
    """

    model_config = ConfigDict(
        extra="allow",
        serialize_by_alias=True,
    )

    data_image_keys: list[str] = Field(
        default=["observation.images.main", "observation.images.secondary_0"],
        description="Comma-separated list of image keys to use for training",
        serialization_alias="data.image_keys",
    )
    save_interval: int = Field(
        default=100,
        description="Number of steps to save the model.",
        gt=0,
    )
    num_train_steps: int = Field(
        default=500,
        description="Number of training steps.",
        gt=0,
        le=3000,
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for training. Decrease it if you get an Out Of Memory (OOM) error",
        gt=0,
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
        gt=0,
    )


class TrainingParamsGr00T(BaseModel):
    class Config:
        extra = "allow"

    validation_dataset_name: Optional[str] = Field(
        default=None,
        description="Optional dataset repository ID on Hugging Face to use for validation",
    )

    batch_size: Optional[int] = Field(
        default=64,
        description="Batch size for training. Decrease it if you get an Out Of Memory (OOM) error",
        gt=0,
        le=128,
        serialization_alias="batch-size",
    )
    num_epochs: int = Field(
        default=10,
        description="Number of epochs to train for.",
        gt=0,
        le=100,
        serialization_alias="num-epochs",
    )
    save_steps: int = Field(
        default=3_000,
        description="Number of steps between saving the model.",
        gt=0,
        le=100_000,
        serialization_alias="save-steps",
    )
    learning_rate: float = Field(
        default=0.0001,
        description="Learning rate for training.",
        gt=0,
        le=1,
    )

    data_dir: str = Field(
        default="data/", description="The directory to save the dataset to"
    )
    validation_data_dir: Optional[str] = Field(
        default=None,
        description="Optional directory to save the validation dataset to. If None, validation is not run.",
    )
    output_dir: str = Field(
        default="outputs/", description="The directory to save the model to"
    )


class TrainingParamsSmolVLA(BaseModel):
    """
    Training parameters are left to None by default and are set depending on the dataset in the training pipeline.
    """

    class Config:
        extra = "allow"

    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for training, we run this on an A10G. Leave it to None to auto-detect based on your dataset",
        gt=0,
        le=150,
    )
    steps: Optional[int] = Field(
        default=None,
        description="Number of training steps. Leave it to None to auto-detect based on your dataset",
        gt=0,
        le=1_000_000,
    )
    save_freq: int = Field(
        default=5_000,
        description="Number of steps between saving the model.",
        gt=0,
        le=1_000_000,
    )


class BaseTrainerConfig(BaseModel):
    model_type: Literal["ACT", "ACT_BBOX", "gr00t", "pi0.5", "smolvla", "custom"] = (
        Field(
            ...,
            description="Type of model to train, supports 'ACT', 'gr00t', 'pi0.5', and 'smolvla'",
        )
    )
    dataset_name: str = Field(
        ...,
        description="Dataset repository ID on Hugging Face, should be a public dataset",
    )

    model_name: Optional[str] = Field(
        default=None,
        description="Name of the trained model to upload to Hugging Face, should be in the format phospho-app/<model_name> or <model_name>",
    )
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="WandB API key for tracking training, you can find it at https://wandb.ai/authorize",
    )
    training_params: Optional[
        TrainingParamsAct
        | TrainingParamsActWithBbox
        | TrainingParamsGr00T
        | TrainingParamsPi05
        | TrainingParamsSmolVLA
    ] = Field(
        default=None,
        description="Training parameters for the model.",
    )


class TrainingRequest(BaseTrainerConfig):
    """
    Pydantic model for training request validation.
    This version consolidates all model name and parameter logic into a single
    validator to prevent redundant operations and fix the duplicate suffix bug.
    """

    private_mode: bool = Field(
        default=False,
        description="Whether to use private training (PRO users only)",
    )
    user_hf_token: Optional[str] = Field(
        default=None,
        description="User's personal HF token for private training",
    )

    @model_validator(mode="before")
    @classmethod
    def prepare_model_and_params(cls, data: dict) -> dict:
        """
        Consolidated validator that handles all pre-validation logic:
        1. Validates `model_type` and initializes `training_params`.
        2. Determines the correct namespace (public or private).
        3. Implements the new logic for handling or generating the model name.
        """
        # Step 1: Validate model_type and training_params
        model_type_to_class: dict[str, type[BaseModel]] = {
            "ACT": TrainingParamsAct,
            "ACT_BBOX": TrainingParamsActWithBbox,
            "gr00t": TrainingParamsGr00T,
            "pi0.5": TrainingParamsPi05,
            "smolvla": TrainingParamsSmolVLA,
        }
        model_type = data.get("model_type")
        if not model_type:
            raise ValueError("Model type is required.")
        if model_type not in model_type_to_class:
            raise ValueError(f"Unsupported model type: {model_type}.")

        params_class = model_type_to_class[model_type]
        if "training_params" in data and data["training_params"]:
            data["training_params"] = params_class.model_validate(
                data["training_params"]
            )
        else:
            data["training_params"] = params_class()  # Set defaults

        # Step 2: Determine the namespace and validate user token
        namespace = "phospho-app"
        api = None
        if data.get("private_mode"):
            token = data.get("user_hf_token") or get_hf_token()
            if not token:
                raise ValueError("Private training requires a valid HF token.")
            data["user_hf_token"] = token
            try:
                api = HfApi(token=token)
                user_info = api.whoami()
                username = user_info.get("name")
                if not username:
                    raise ValueError("Could not get username from HF token.")
                namespace = username
            except Exception as e:
                raise ValueError(f"Failed to validate user namespace: {e}")

        # Step 3 & 4: Handle model name based on the new business rules
        user_provided_name = data.get("model_name")
        max_len = 96  # Max length for a model name

        if user_provided_name:
            # Rule 1: If a model name is provided, use it.
            # The base name is the user-provided name.
            base_name = user_provided_name

            # If the namespace is different from phospho-app and not in private mode, raise error
            if namespace != "phospho-app" and not data.get("private_mode"):
                raise ValueError(
                    "You can only use a custom model name in private mode or with the phospho-app namespace."
                )

            # Ensure the name is less than the max length
            if len(base_name) > max_len:
                base_name = base_name[:max_len]

            data["model_name"] = base_name

        else:
            # If no model name is provided, generate one.
            dataset_name = data.get("dataset_name")
            if not dataset_name or len(dataset_name.split("/")) != 2:
                raise ValueError(
                    "A valid dataset_name is required to generate a model name."
                )
            dataset_base_name = dataset_name.split("/")[1]
            base_name = f"{model_type}-{dataset_base_name}"

            random_chars = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )

            # Clamp the base_name length to ensure the final name is valid
            max_base_len = (
                max_len - len(namespace) - len(random_chars) - 2
            )  # for "/" and "-"
            clamped_base_name = base_name[:max_base_len]

            data["model_name"] = f"{namespace}/{clamped_base_name}-{random_chars}"

        return data

    @model_validator(mode="after")
    def validate_dataset_access(self) -> "TrainingRequest":
        """Validate dataset access based on private mode (no changes needed here)."""
        if self.private_mode:
            if not self.user_hf_token:
                raise ValueError("Private training requires a user HF token.")
            try:
                api = HfApi(token=self.user_hf_token)
                api.repo_info(repo_id=self.dataset_name, repo_type="dataset")
            except Exception as e:
                raise ValueError(f"Cannot access dataset {self.dataset_name}: {e}")
        else:
            try:
                url = (
                    f"https://huggingface.co/api/datasets/{self.dataset_name}/tree/main"
                )
                response = requests.get(url, timeout=5)
                response.raise_for_status()
            except Exception:
                raise ValueError(
                    f"Dataset {self.dataset_name} is not a valid, public dataset."
                )

        return self


class HuggingFaceTokenValidator:
    @staticmethod
    def has_write_access(hf_token: str, hf_model_name: str, private: bool) -> bool:
        """Check if the HF token has write access by attempting to create a repo."""
        api = HfApi(token=hf_token)
        try:
            api.create_repo(
                hf_model_name, private=private, exist_ok=True, token=hf_token
            )
            return True  # The token has write access
        except Exception as e:
            print(f"Write access check failed: {e}")
            return False  # The token does not have write access


def generate_readme(
    model_type: str,
    dataset_repo_id: str,
    training_params: BaseModel,
    folder_path: Optional[Path] = None,
    wandb_run_url: Optional[str] = None,
    error_traceback: Optional[str] = None,
    return_readme_as_bytes: bool = False,
) -> Path | bytes:
    readme = f"""
---
datasets: {dataset_repo_id}
library_name: phosphobot
pipeline_tag: robotics
model_name: {model_type}
tags:
- phosphobot
- {model_type}
task_categories:
- robotics
---

# {model_type} model - 🧪 phosphobot training pipeline

- **Dataset**: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})
- **Wandb run id**: {wandb_run_url}

"""
    if error_traceback:
        readme += f"""
## Error Traceback
We faced an issue while training your model.

```
{error_traceback}
```

"""
    else:
        readme += """
## This model was trained using **[🧪phospho](https://phospho.ai)**

Training was successful, try it out on your robot!

"""

    readme += f"""
## Training parameters

```text
{training_params.model_dump_json(exclude_unset=True, by_alias=True, indent=2)}
```

📖 **Get Started**: [docs.phospho.ai](https://docs.phospho.ai?utm_source=huggingface_readme)

🤖 **Get your robot**: [robots.phospho.ai](https://robots.phospho.ai?utm_source=huggingface_readme)
"""
    if return_readme_as_bytes:
        return readme.encode("utf-8")
    if folder_path is not None:
        readme_path = folder_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)
        return readme_path
    else:
        raise ValueError(
            "folder path is None and return_readme_as_bytes is False. Please provide a valid folder path. If you want to return the readme as bytes, set return_readme_as_bytes to True."
        )


def resize_dataset(
    dataset_root_path: Path,
    resize_to: tuple = (320, 240),
) -> tuple[bool, bool, Optional[str]]:
    """
    Resize the dataset to a smaller size for faster training.

    Args:
        dataset_root_path (Path): Path to the dataset root directory.

    Returns:
        1st bool: True if the processing was successful, False otherwise.
        2nd bool: True if we need to recompute the stats, False otherwise.
        str: Details if error
    """
    # Start by opening the InfoModel and checking the video sizes
    logger.info(
        f"Resizing videos in {dataset_root_path} to {resize_to[0]}x{resize_to[1]}"
    )
    try:
        meta_path = dataset_root_path / "meta"
        video_information = {}
        validated_info_model = InfoModel.from_json(
            meta_folder_path=str(meta_path.resolve())
        )
        for feature in validated_info_model.features.observation_images:
            shape = validated_info_model.features.observation_images[feature].shape
            if shape != [resize_to[1], resize_to[0], 3]:
                video_information[feature] = {
                    "need_to_resize": True,
                    "shape": shape,
                }
                validated_info_model.features.observation_images[feature].shape = [
                    resize_to[1],
                    resize_to[0],
                    3,
                ]
            else:
                logger.info(f"Video {feature} is already in the correct size {shape}")

        if video_information == {}:
            logger.info("No videos need to be resized.")
            return True, False, "No videos need to be resize"

        for video_folder in video_information:
            if video_information[video_folder]["need_to_resize"]:
                video_path = dataset_root_path / "videos" / "chunk-000" / video_folder
                for episode in video_path.iterdir():
                    if episode.suffix == ".mp4" and not episode.name.startswith(
                        "edited_"
                    ):
                        out_path = episode.parent / f"edited_{episode.name}"

                        # Open input video
                        input_container = av.open(str(episode))
                        input_stream = input_container.streams.video[0]

                        # Open output video
                        output_container = av.open(str(out_path), mode="w")
                        output_stream = output_container.add_stream(
                            codec_name="h264",
                            rate=input_stream.base_rate,
                        )
                        output_stream.width = resize_to[0]  # type: ignore
                        output_stream.height = resize_to[1]  # type: ignore
                        output_stream.pix_fmt = input_stream.pix_fmt  # type: ignore

                        # Process frames
                        for frame in input_container.decode(video=0):
                            # Resize frame
                            frame = frame.reformat(
                                width=resize_to[0],
                                height=resize_to[1],
                            )

                            # Encode frame
                            packet = output_stream.encode(frame)  # type: ignore
                            output_container.mux(packet)

                        # Flush encoder
                        for value in output_stream.encode(None):  # type: ignore
                            output_container.mux(value)

                        input_container.close()
                        output_container.close()

                # Remove original videos and rename edited ones
                for episode in video_path.iterdir():
                    if episode.suffix == ".mp4" and episode.name.startswith("edited_"):
                        new_name = episode.name.replace("edited_", "")
                        new_path = episode.parent / new_name
                        new_path.unlink(missing_ok=True)
                        episode.rename(new_path)

        # Save updated info.json
        validated_info_model.to_json(meta_folder_path=str(meta_path.resolve()))

        logger.info("Resizing completed.")
        logger.warning("You now need to recompute the stats for the dataset.")
        return True, True, "Resizing successful"

    except Exception as e:
        logger.error(f"Error resizing videos: {e}")
        return False, False, f"Error resizing videos: {e}"


class BaseTrainer(ABC):
    """
    Abstract class for training models.
    """

    @abstractmethod
    def train(self) -> None:
        pass
