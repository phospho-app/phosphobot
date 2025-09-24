# Makefile for running simulation and launching server


# Default output filename when running `make build`
OUTPUT_FILENAME ?= main.bin

# Default target
all: prod

# Run the server for prod settings (able to connect to the Meta Quest). If npm is not installed, it will skip the build step.
prod:
	cd ./dashboard && (npm i && npm run build && mkdir -p ../phosphobot/resources/dist/ && cp -r ./dist/* ../phosphobot/resources/dist/)
	cd phosphobot && uv run --python 3.10 phosphobot run --simulation=headless --no-crash-telemetry

prod_back:
	cd phosphobot && MESA_GL_VERSION_OVERRIDE=3.3 uv run --python 3.10 phosphobot run --simulation=headless --no-crash-telemetry

# Chat agent run
chat:
	cd phosphobot && uv run --python 3.10 phosphobot run --chat --simulation=headless --no-crash-telemetry

# Same as prod, but with the simulation GUI (useful when adding new robots to test)
prod_gui:
	cd ./dashboard && (npm i && npm run build && mkdir -p ../phosphobot/resources/dist/ && cp -r ./dist/* ../phosphobot/resources/dist/)
	cd phosphobot && MESA_GL_VERSION_OVERRIDE=3.3 uv run --python 3.10 phosphobot run --simulation=gui --no-crash-telemetry

# Trick for raspberrypi : pretend opengl>=3.2 so that pybullet will run in a gui
prod_gui_back:
	cd phosphobot && MESA_GL_VERSION_OVERRIDE=3.3 uv run --python 3.10 phosphobot run --simulation=gui --no-crash-telemetry


# Run the server for prod settings (able to connect to the Meta Quest) but with telemetry disabled. If npm is not installed, it will skip the build step.
prod_no_telemetry:
	cd ./dashboard && ((npm i && npm run build && mkdir -p ../phosphobot/resources/dist/ && cp -r ./dist/* ../phosphobot/resources/dist/) || echo "npm command failed, continuing anyway") 
	cd phosphobot && uv run phosphobot run --simulation=headless --no-telemetry

# Run the server for prod settings with the simulation enabled
prod_sim:
	cd ./phosphobot && uv run phosphobot run --simulation=gui --no-telemetry

# Run info command for prod settings
prod_info:
	cd ./phosphobot && uv run phosphobot info --opencv --servos

# Run localhost server for dev settings
local:
	cd ./phosphobot && uv run phosphobot run --simulation=gui --port=8080 --host=127.0.0.1 --no-telemetry

# For running integration tests
test_server:
	cd ./phosphobot && uv run phosphobot run --simulation=headless --only-simulation --simulate-cameras --port=8080 --host=127.0.0.1 --no-telemetry &

# For running the built app in test mode
run_bin_test:
	./phosphobot/dist/main.bin run --simulation=headless --simulate-cameras --port=8080 --host=127.0.0.1 --no-telemetry &

# For running the built app in prod mode
run_bin:
	./phosphobot/dist/main.bin run --simulation=headless

# To have the information about the software and the hardware from the built app
info_bin:
	./phosphobot/dist/main.bin info


# Don't use sudo uv run, it will break CICD
build_pyinstaller:
	cd phosphobot && \
	WASMTIME_PATH=$$(uv run python -c "import wasmtime; import os; print(os.path.dirname(wasmtime.__file__))") && \
	uv run pyinstaller \
	--onefile \
	--name $(OUTPUT_FILENAME) \
	--add-data "resources:resources" \
	--add-data "$$WASMTIME_PATH:wasmtime" \
	--additional-hooks-dir "./hooks" \
	--hidden-import phosphobot \
	--collect-all phosphobot \
	--collect-all wasmtime \
	--clean -c \
	phosphobot/main.py \

clean_build:
	cd ./phosphobot && rm -rf main.build main.dist main.onefile-build $(OUTPUT_FILENAME)

build_frontend:
	cd ./dashboard && npm i && npm run build && \
	mkdir -p ../phosphobot/resources/dist/ && \
	cp -r ./dist/* ../phosphobot/resources/dist/

# Clean up
stop:
	echo "Checking for uv process..."
	ps -ef | grep uv || echo "No uv process running."
	echo "Checking for Python processes..."
	ps -ef | grep python || echo "No Python processes found."
	-killall uv || echo "No uv process found."
	-killall python3 || echo "No python3 process found."
	-killall python3.8 || echo "No python3.8 process found."
	echo "Cleanup complete."

# Clean up hard (kill processes on ports 80 and 8080)
stop_hard:
	echo "Killing all processes listening on port 80..."
	-sudo lsof -i :80 -t | xargs -r sudo kill -9
	echo "Killing all processes listening on port 8020..."
	-sudo lsof -i :8020 -t | xargs -r sudo kill -9
	echo "Killing all processes listening on port 8080..."
	-sudo lsof -i :8080 -t | xargs -r sudo kill -9
	echo "Forceful cleanup of all processes on ports 80, 8020 and 8080 complete."


submodule:
	git submodule update --init --recursive

types: 
	cd phosphobot && uv run mypy . --check-untyped-defs --disallow-untyped-defs --ignore-missing-imports --follow-imports=silent

sort:
	cd phosphobot && uv run ruff check --select I --fix .

tests:
	cd phosphobot && uv run pytest tests/phosphobot/ -n 5


.PHONY: all dev prod prod_gui stop stop_hard dataset_annotate dataset_convert dataset_push robot_watch test_server build clean_build build_pyinstaller run_bin run_bin_test info_bin