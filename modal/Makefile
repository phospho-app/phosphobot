install:
	uv sync && uv run modal setup && uv run modal config set-environment test

deploy_admin:
	cd admin && uv run modal deploy app.py

deploy_act:
	cd act && uv run modal deploy -m src.app

deploy_paligemma:
	cd paligemma && uv run modal deploy app.py

deploy_gr00t:
	cd gr00t && uv run modal deploy -m src.app

deploy_pi0:
	cd pi0 && uv run modal deploy -m src.app

deploy_all:
	make deploy_admin && make deploy_act && make deploy_gr00t && make deploy_paligemma && make deploy_pi0

PHONY: deploy_admin deploy_act deploy_paligemma deploy_gr00t deploy_pi0