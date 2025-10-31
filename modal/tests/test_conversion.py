import modal
from huggingface_hub import HfApi

dataset_conversion_from_v2 = modal.Function.from_name(
    "conversion_app_from_v2", "convert_dataset_to_v21"
)
dataset_conversion_from_v21 = modal.Function.from_name(
    "conversion_app_from_v21", "convert_dataset_to_v3"
)

dataset_name = "phospho-app/tomato_bboxes"
token = None

api = HfApi(token=token)

# Delete old v3.0 branch of dataset if it exists
try:
    api.delete_tag(
        repo_id=dataset_name,
        tag="v3.0",
        token=token,
        repo_type="dataset",
    )
    print(f"Deleted old v3.0 branch of dataset {dataset_name}")
except Exception:
    pass


def test_dataset_conversion():
    dataset, error_str = dataset_conversion_from_v2.remote(
        dataset_name=dataset_name,
        huggingface_token=token,
    )
    if error_str is None:
        dataset, error_str = dataset_conversion_from_v21.remote(
            dataset_name=dataset_name,
            huggingface_token=token,
        )
    print(f"Converted dataset: {dataset}, error: {error_str}")


if __name__ == "__main__":
    test_dataset_conversion()
