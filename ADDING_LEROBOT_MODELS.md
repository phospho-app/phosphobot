# Adding Support for New LeRobot Policy Models

This guide explains how to add support for a new LeRobot policy model to phosphobot. Phosphobot currently supports [ACT (Action Chunking Transformer)](https://arxiv.org/abs/2304.13705) and [SmolVLA](https://arxiv.org/abs/2506.01844) models, and follows a consistent pattern that makes adding new models easy and straightforward. This architecture ensures consistent behavior across all LeRobot policy models while allowing for model-specific customizations and optimizations.

## Architecture Overview

Adding support for serving and training (fine-tuning) a new LeRobot policy requires changes across the following parts of the framework:

### Backend
1. **Core Action Model Layer** (`phosphobot/am/`): Client-side model interfaces and control logic, model-specific validators, and spawn configurations
2. **Modal Infrastructure Layer** (`modal/lerobot_modal/`): Server-side inference and training infrastructure
3. **Phospho Dashboard**: AI Control and AI Training endpoints

### Frontend
1. **Phospho Dashboard** (`dashboard/src`): AI Control and AI Training endpoints

### CI/CD
1. **Deployment of policies** (`.github/workflows`): Deploy on commits pushed to main

## A typical implementation workflow

### Step 1: Backend Changes

#### 1. Create the Core Action Model Class

Create a new file `phosphobot/am/your_policy.py` and implement the following:

1. **Create config validator dataclasses:**

```python
class YourPolicyHuggingFaceModelValidator(HuggingFaceModelValidator):
    type: Literal["model_type"]


class YourPolicyHuggingFaceAugmentedValidator(HuggingFaceAugmentedValidator):
    type: Literal["model_type"]


class YourPolicySpawnConfig(LeRobotSpawnConfig):
    hf_model_config: YourPolicyHuggingFaceAugmentedValidator  # type: ignore[assignment]
```

2. **Inherit a child policy class from `LeRobot` and add policy-specific config validation and input preparation logic:**

```python
class YourPolicy(LeRobot):
    """Action model implementation for YourPolicy. Inherits from LeRobot."""

    @classmethod
    def _get_model_validator_class(cls) -> type:
        """Return YourPolicy-specific model validator class"""
        return YourPolicyHuggingFaceModelValidator

    @classmethod
    def _get_augmented_validator_class(cls) -> type:
        """Return YourPolicy-specific augmented validator class"""
        return YourPolicyHuggingFaceAugmentedValidator

    @classmethod
    def _get_spawn_config_class(cls) -> type:
        """Return YourPolicy-specific spawn config class"""
        return YourPolicySpawnConfig

    def _prepare_model_inputs(
        self,
        # params
    ) -> Dict[str, np.ndarray | str]:
        """Prepare model inputs for YourPolicy"""
        # Add state and image inputs
        inputs: Dict[str, np.ndarray | str] = {
            config.input_features.state_key: state,
            **image_inputs,
        }

        # Add model-specific input processing here
        # Examples:
        # - For prompt-based models (like SmolVLA): add prompt
        # - For object detection models (like ACT): add detection instructions
        # - For other modalities: add custom preprocessing

        return inputs
```

#### 2. Define Training Params for your policy

1. **Add a new `TrainingParams` class in `phosphobot/am/base.py`**:

```python
class TrainingParamsYourPolicy(BaseModel):
    # define params and set default values
    batch_size: Optional[int]
    steps: Optional[int]
    save_freq: int
    # more training params
```

2. **Update `BaseTrainerConfig` class to consider training params for new policy:**

#### 3. Create the Modal Endpoint

Create a new file `modal/lerobot_modal/your_policy.py` and implement the following:

1. **Import common functions from `lerobot_modal/helper.py` and `lerobot_modal/app.py`**
```python
from phosphobot.am.yourpolicy import YourPolicySpawnConfig
from .helper import validate_inputs, prepare_base_batch
from .app import base_image, phosphobot_dir, serve_policy, train_policy
```

2. **Create a modal image, a modal app, and a modal volume for your policy**
- Modal Image: Container Image deployed on Modal
    - Use the installation / Docker deployment instructions from the policy source repo to create this
    - Use Method Chaining, Layer Caching, and other best practices suggested by Modal (Ref: https://modal.com/docs/guide/images)
- Modal App: Used to deploy serverless functions for serving and training your policy model
- Modal Volume: Persistent volume to store frequently used data (model weights, datasets) to reduce latency

```python
your_policy_image = (
    base_image.uv_pip_install(
        "lerobot[your_policy]==<version>",
    )
    .pip_install_from_pyproject(pyproject_toml=str(phosphobot_dir / "pyproject.toml"))
    .add_local_python_source("phosphobot")
)

your_policy_app = modal.App("your_policy-server")
your_policy_volume = modal.Volume.from_name("your_policy", create_if_missing=True)
```

3. **[Optional] Add any additional functions required for inference / training of the new policy**

4. **Add inference implementation**
```python
def process_yourpolicy_inference():
    # Validate inputs using common function
    validate_inputs(current_qpos, images, model_specifics, target_size)

    # Add model-specific validation here
    # Example:
    # if some_required_param is None:
    #     raise ValueError("some_required_param cannot be None for YourPolicy inference")

    with torch.no_grad(), torch.autocast(device_type="cuda"):
        # Prepare base observation using common function
        batch = prepare_base_batch(current_qpos, images, image_names, model_specifics, target_size)

        # Add model-specific batch preparation here
        # Examples:
        # - Add prompts: batch["task"] = prompt
        # - Add environment state: batch[model_specifics.env_key] = env_state
        # - Add custom preprocessing: batch["custom_key"] = custom_data

        # Run policy inference and get actions
        actions = policy.predict_action_chunk(batch)

        return actions.cpu().numpy()    # expected shape = [1, ACTION_HORIZON, ACTION_SHAPE]
```

5. **Add Modal function for serving trained model**
```python
@your_policy_app.function(
    image=your_policy_image,
    # other args
)
async def serve():
    """YourPolicy model serving function."""
    await serve_policy()
```

6. **Add Modal function for training model**
```python
@your_policy_app.function(
    image=your_policy_image,
    # other args
)
async def train():
    """YourPolicy training function."""
    # remove model_type from kwargs if present
    # to avoid passing model_type twice in the `train_policy()` function call
    if "model_type" in kwargs:
        del kwargs["model_type"]

    train_policy(
        model_type="smolvla"
        # other args
    )
```

#### 4. Update common LeRobot inference and training logic

In `modal/lerobot_modal/app.py`, add your policy's specific inference and training logic:

1. **Import your model config and training params class:**
```python
from phosphobot.am.yourpolicy import YourPolicySpawnConfig
from phosphobot.am.base import TrainingParamsYourPolicy
```

2. **Add to supported model types:**
```python
_SUPPORTED_MODEL_TYPES = ["act", "smolvla", "yourpolicy"]
```

3. **Update the `serve_policy` function** to handle your policy:
```python
async def serve_policy():
from .your_policy.py import process_yourpolicy_inference

    @app.post("/act")
    async def inference(request: InferenceRequest):

        if isinstance(model_specifics, YourPolicySpawnConfig):
            actions = process_yourpolicy_inference()
```

4. **Update typing suggestions to avoid mypy errors**


#### 5. Make changes to enable AI control and training via phosphobot dashboard

Files to update:
- `phosphobot/phosphobot/endpoints/pages.py`
- `phosphobot/phosphobot/models/__init__.py`
- `phosphobot/phosphobot/ai_control.py`

Make appropriate imports and update the above files. For reference, look at the implementation for already supported policies.

### Step 2: Frontend Changes

1. `dashboard/src/lib/hooks.ts`: Add the new policy as `modelType` to the React Global Store so it's available across different pages.
2. `dashboard/src/pages/AIControlPage.tsx`: Add new policy, optionally add policy to `modelsThatRequirePrompt` if applicable.
3. `dashboard/src/pages/AITrainingPage.tsx`: Add new policy

### Step 3: Github Workflow Changes

Update `.github/workflows/deploy_modal.yml` to deploy the new policy on Modal for CI / CD.
