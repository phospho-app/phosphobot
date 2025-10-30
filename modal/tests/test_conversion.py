import modal
from huggingface_hub import HfApi

dataset_conversion = modal.Function.from_name("conversion-app", "convert_dataset_to_v3")

dataset_name = "LegrandFrederic/pick_and_place"
token = "to_fill"

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
    dataset, error_str = dataset_conversion.remote(
        dataset_name=dataset_name,
        huggingface_token=None,
    )
    print(f"Converted dataset: {dataset}, error: {error_str}")


if __name__ == "__main__":
    test_dataset_conversion()
