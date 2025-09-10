import modal
import numpy as np
from PIL import Image
import requests

paligemma_warmup = modal.Function.from_name("paligemma-detector", "warmup_model")
paligemma_detect = modal.Function.from_name("paligemma-detector", "detect_object")

BATCH_SIZE = 1


def test_paligemma_warmup():
    paligemma_warmup.spawn()


def test_paligemma_detect():
    paligemma_warmup.spawn()

    # This is a picture of a blue car
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)

    # Reshape the image to (224, 224, 3)
    image = image.resize((224, 224))

    # Make it a numpy array with shape (100, 224, 224, 3)
    frames = np.array([np.array(image)] * BATCH_SIZE, dtype=np.uint8)

    print(frames.shape)

    bboxes = paligemma_detect.remote(
        frames=frames,
        instructions=["a car"] * BATCH_SIZE,
    )

    assert bboxes != [[0.0, 0.0, 0.0, 0.0]] * BATCH_SIZE, (
        f"Bboxes not detected: {bboxes}"
    )
    print("Bboxes:", bboxes)


if __name__ == "__main__":
    test_paligemma_detect()
