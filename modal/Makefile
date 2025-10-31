install:
	uv sync && uv run modal setup && uv run modal config set-environment test

deploy_admin:
	cd admin && uv run modal deploy -m src.app

deploy_act:
	uv run modal deploy -m lerobot_modal.act::act_app

deploy_smolvla:
	uv run modal deploy -m lerobot_modal.smolvla::smolvla_app

deploy_conversion:
	uv run modal deploy -m lerobot_modal.conversion_service_from_v2::conversion_service_from_v2 && uv run modal deploy -m lerobot_modal.conversion_service_from_v21::conversion_service_from_v21

deploy_paligemma:
	cd paligemma && uv run modal deploy app.py

deploy_gr00t:
	cd gr00t && uv run modal deploy -m src.app

deploy_pi05:
	cd pi05 && uv run modal deploy app.py

deploy_vllm:
	cd vllm && uv run modal deploy app.py

deploy_all:
	make deploy_admin && make deploy_act && make deploy_gr00t && make deploy_paligemma && make deploy_vllm && make deploy_pi05 && make deploy_smolvla && make deploy_conversion

PHONY: deploy_all
