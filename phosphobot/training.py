import json
from loguru import logger
from pydantic import BaseModel
from phosphobot.utils import get_home_app_path

phosphobot_training_json = get_home_app_path() / "training.json"


class ModelTrainingConfig(BaseModel):
    model_name: str
    dataset_name: str
    url: str
    creation_date: str
    model_type: str


class TrainingConfig(BaseModel):
    models: list[ModelTrainingConfig]


def get_training_config() -> TrainingConfig:
    """Get the training configuration from the JSON file."""
    try:
        with open(phosphobot_training_json, "r") as f:
            loaded_config = json.load(f)
            validated_config = TrainingConfig(**loaded_config)
        return validated_config
    except FileNotFoundError:
        # Create an empty config if the file does not exist
        with open(phosphobot_training_json, "w") as f:
            json.dump({"models": []}, f, indent=4)
        return TrainingConfig(models=[])
    except json.JSONDecodeError:
        # Handle JSON decode error
        logger.warning(
            f"Error decoding JSON from {phosphobot_training_json}. Creating a new empty file."
        )
        with open(phosphobot_training_json, "w") as f:
            json.dump({"models": []}, f, indent=4)
        return TrainingConfig(models=[])
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return TrainingConfig(models=[])


def add_to_training_config(new_model: ModelTrainingConfig) -> TrainingConfig:
    """Add a new model to the training configuration."""
    config = get_training_config()

    # Check if the model already exists and update it if it does
    for i, model in enumerate(config.models):
        if model.model_name == new_model.model_name:
            config.models[i] = new_model
            break
    else:
        config.models.insert(0, new_model)

    with open(phosphobot_training_json, "w") as f:
        json.dump(config.model_dump(), f, indent=4)

    return config


def delete_model_from_training_config(model_name: str) -> TrainingConfig:
    """Delete a model from the training configuration."""
    config = get_training_config()
    config.models = [model for model in config.models if model.model_name != model_name]

    with open(phosphobot_training_json, "w") as f:
        json.dump(config.model_dump(), f, indent=4)

    return config
