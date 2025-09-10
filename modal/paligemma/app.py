import re
import os
import modal
import numpy as np
from PIL import Image
from loguru import logger
from typing import List

# Modal image with all necessary dependencies
paligemma_image = modal.Image.debian_slim(python_version="3.13").uv_pip_install(
    "accelerate>=1.7.0",
    "pyarrow>=20.0.0",
    "torch>=2.7.0",
    "transformers>=4.52.3",
    "packaging",
    "loguru",
    "pillow",
)

app = modal.App("paligemma-detector")
paligemma_volume = modal.Volume.from_name("PaliGemma", create_if_missing=True)

MINUTE = 60  # seconds

FUNCTION_IMAGE = paligemma_image
FUNCTION_GPU: list[str | modal.gpu._GPUConfig | None] = ["T4"]
FUNCTION_SCALEDOWN_WINDOW = 1 * MINUTE
FUNCTION_TIMEOUT = 5 * MINUTE


class ObjectDetectionProcessor:
    def __init__(self, model_path: str = "google/paligemma2-3b-mix-224"):
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        logger.info(f"Loading PaliGemma model: {model_path}")

        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            logger.error("Hugging Face token not found in environment variables.")
            raise ValueError("Hugging Face token is required for model access.")

        cache_dir = "/data/paligemma2-3b-mix-224"

        # Load processor & model
        self.processor = AutoProcessor.from_pretrained(
            model_path, token=hf_token, cache_dir=cache_dir
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            cache_dir=cache_dir,
        )
        self.device = "cuda"
        self.dtype = torch.float16

        self.model.eval()
        logger.info(f"Model loaded on device: {self.device}")

    def parse_bbox(self, text: str) -> list[float]:
        """Parse bounding box coordinates from model output text."""
        # Find four <loc####> tokens
        locs = re.findall(r"<loc(\d{3,4})>", text)
        if len(locs) >= 4:
            y1, x1, y2, x2 = map(int, locs[:4])
            # Normalize to [0,1] range
            bbox = [x1 / 1024, y1 / 1024, x2 / 1024, y2 / 1024]
            logger.info(f"Parsed bbox: {bbox}")
            return bbox

        logger.warning(
            f"Failed to parse bbox from text: {text}. Expected four <loc####> tokens."
        )
        return [0.0, 0.0, 0.0, 0.0]

    def detect(
        self, images: List[Image.Image], instructions: List[str]
    ) -> List[List[float]]:
        """Detect object in image based on instruction."""
        import torch

        assert len(images) == len(instructions), (
            f"Number of images ({len(images)}) and instructions ({len(instructions)}) must be the same"
        )

        try:
            for image in images:
                # Ensure image is 224x224 as required
                if image.size != (224, 224):
                    image = image.resize((224, 224), Image.Resampling.LANCZOS)

                # Convert to RGB if necessary
                if image.mode != "RGB":
                    image = image.convert("RGB")

            # Prepare prompt
            prompts = [f"<image> detect {instruction}" for instruction in instructions]

            # Process inputs
            inputs = self.processor(
                text=prompts, images=images, return_tensors="pt"
            ).to(self.device)

            input_length = inputs["input_ids"].shape[-1]

            # Generate prediction
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_length=input_length + 50,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    disable_compile=True,
                )

            # Decode output
            decoded = [
                self.processor.decode(output[input_length:], skip_special_tokens=True)
                for output in outputs
            ]

            return [self.parse_bbox(output) for output in decoded]

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return [[0.0, 0.0, 0.0, 0.0]]


detector: ObjectDetectionProcessor | None = None


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    scaledown_window=FUNCTION_SCALEDOWN_WINDOW,
    timeout=FUNCTION_TIMEOUT,
    secrets=[modal.Secret.from_name("huggingface")],
)
def warmup_model() -> None:
    """Warm up the model by loading it into memory."""
    global detector
    if detector is None:
        detector = ObjectDetectionProcessor()


@app.function(
    image=FUNCTION_IMAGE,
    gpu=FUNCTION_GPU,
    scaledown_window=FUNCTION_SCALEDOWN_WINDOW,
    timeout=FUNCTION_TIMEOUT,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": paligemma_volume},
)
def detect_object(frames: np.ndarray, instructions: List[str]) -> List[List[float]]:
    """
    Detect objects in a list of frames using PaliGemma.

    Args:
        frames: frames to detect objects shape (B, H, W, 3) RGB expected.
        instructions: List of instructions to use for object detection.

    Returns:
        List of bounding boxes for each frame. Shape (B, 4)
    """
    global detector
    if detector is None:
        detector = ObjectDetectionProcessor()

    try:
        if not isinstance(frames, np.ndarray):
            raise ValueError("Expected a NumPy array for `frame`.")
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(
                f"Expected a uint8 RGB array with shape (B, H, W, 3), got dtype={frames.dtype}, shape={frames.shape}"
            )

        images = [Image.fromarray(frame) for frame in frames]
    except Exception as e:
        logger.error(f"Failed to read images: {e}")
        return [[0.0, 0.0, 0.0, 0.0] for _ in range(len(frames))]

    # Perform detection
    bboxes = detector.detect(images, instructions)

    return bboxes
