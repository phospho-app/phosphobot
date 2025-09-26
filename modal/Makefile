install:
	uv sync && uv run modal setup && uv run modal config set-environment test

deploy_admin:
	cd admin && uv run modal deploy -m src.app

deploy_act:
	uv run modal deploy -m lerobot_modal.act::act_app

deploy_smolvla:
	uv run modal deploy -m lerobot_modal.smolvla::smolvla_app

deploy_paligemma:
	cd paligemma && uv run modal deploy app.py

deploy_gr00t:
	cd gr00t && uv run modal deploy -m src.app

deploy_pi05:
	cd pi05 && uv run modal deploy app.py

deploy_vllm:
	cd vllm && uv run modal deploy app.py

deploy_all:
	make deploy_admin && make deploy_act && make deploy_gr00t && make deploy_paligemma && make deploy_vllm && make deploy_pi05 && make deploy_smolvla

PHONY: deploy_admin deploy_act deploy_paligemma deploy_gr00t deploy_vllm deploy_all install deploy_pi05 deploy_smolvla
