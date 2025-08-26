"""
Store constants and configurations for the app in this file.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

import yaml  # type: ignore
from loguru import logger
from pydantic import BaseModel, Field

from phosphobot.types import SimulationMode, VideoCodecs
from phosphobot.utils import get_home_app_path

YAML_CONFIG_PATH = str(get_home_app_path() / "config.yaml")


def rename_keys_for_config(config_dict: dict) -> dict:
    """
    Rename the keys to correspond to the fields in the Configuration class.
    We write the keys in uppercase and add DEFAULT_ in front of them.
    """
    return {
        f"DEFAULT_{key.upper()}" if not key.startswith("DEFAULT_") else key: value
        for key, value in config_dict.items()
    }


def remove_default_prefix(config_dict: dict) -> dict:
    """
    Remove the DEFAULT_ prefix from the keys.
    """
    return {
        key.replace("DEFAULT_", "").lower(): value for key, value in config_dict.items()
    }


class Configuration(BaseModel):
    # Server
    PORT: int = 80

    # Profiling (creates a profile.html file in the root directory)
    PROFILE: bool = False

    # Recording
    MAIN_CAMERA_ID: Optional[int] = None  # defaults to min(detected cameras)

    # Whether to initialize the RealSense camera
    ENABLE_REALSENSE: bool = True
    ENABLE_CAMERAS: bool = True
    ENABLE_CAN: bool = True  # Enable CAN scanning
    # Enable crash reporting and usage telemetry
    CRASH_TELEMETRY: bool = False
    USAGE_TELEMETRY: bool = False

    # How simulation should be run
    SIM_MODE: SimulationMode = SimulationMode.headless
    # Only simulation: Only use the simulation
    ONLY_SIMULATION: bool = False
    SIMULATE_CAMERAS: bool = False
    MAX_OPENCV_INDEX: int = 10

    # HF token
    HF_TOKEN_VALID: bool = False

    # Private mode
    DEFAULT_HF_PRIVATE_MODE: bool = False

    # These fields will be set after loading the user config
    DEFAULT_DATASET_NAME: str = "example_dataset"
    DEFAULT_FREQ: int = 30
    DEFAULT_EPISODE_FORMAT: Literal["lerobot_v2.1", "lerobot_v2", "json"] = (
        "lerobot_v2.1"
    )
    DEFAULT_VIDEO_CODEC: VideoCodecs = Field(default_factory=lambda: "avc1")
    DEFAULT_VIDEO_SIZE: List[int] = [320, 240]
    DEFAULT_TASK_INSTRUCTION: str = "None"
    # List of camera ids to disable, set to -1 to disable all cameras
    DEFAULT_CAMERAS_TO_DISABLE: Optional[List[int]] = None
    DEFAULT_CAMERAS_TO_RECORD: Optional[List[int]] = None

    class Config:
        extra = "ignore"

    @classmethod
    def from_yaml(
        cls, config_path: Optional[Union[str, Path]] = None
    ) -> "Configuration":
        """
        Load configuration from a YAML file.
        Args:
            config_path (str): Path to the YAML configuration file.
        Returns:
            dict: Dictionary containing configuration values.
        """
        if config_path is None:
            config_path = YAML_CONFIG_PATH

        # Ensure the file exists. If not, create it.
        if not Path(config_path).exists():
            with open(config_path, "w") as file:
                file.write("")
            return cls()

        with open(config_path, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                logger.error(
                    f"Error loading configuration file: {e}.\nUsing default config. Please shut down the program and edit the config file."
                )
                config = None

        if config is None or not isinstance(config, dict):
            config = {}
        config = rename_keys_for_config(config)

        return cls(**config)

    def save_user_settings(self, user_settings: dict) -> None:
        """
        Save user settings to a YAML file.
        Args:
            user_settings (dict): Dictionary containing user settings.
        """

        try:
            # Read the user_settings as a config
            user_settings = rename_keys_for_config(user_settings)
            new_config = Configuration.model_validate(user_settings)
        except Exception as e:
            logger.error(f"Error saving user settings: {e}")
            return

        # If the user_settings are valid, save them to the YAML file
        with open(YAML_CONFIG_PATH, "w") as file:
            yaml.dump(user_settings, file)

        # Then, replace inplace the fields of the current instance with the new instance
        for field in Configuration.model_fields.keys():
            setattr(self, field, getattr(new_config, field))


config = Configuration.from_yaml()
