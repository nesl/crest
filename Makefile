.PHONY: help install test integration-test test-all start-gpu start-hil smoke env-create arduino-setup stm32-setup prepare-dataset clean

PYTHON ?= python
ENV ?= tinyodomex
OXIOD_ZIP ?= OxIOD.zip

help:
	@echo "Targets:"
	@echo "  install        Install editable package without deps (use conda for deps)"
	@echo "  test           Run the fast/default pytest suite in test/"
	@echo "  integration-test  Run slow integration tests in test/integration/"
	@echo "  test-all       Run both the fast suite and integration tests"
	@echo "  start-gpu      Run NAS client (GPU box)"
	@echo "  start-hil      Run HIL server (device host)"
	@echo "  smoke          Run a short NAS smoke test (override ARGS)"
	@echo "  env-create     Create conda env from environment.yml"
	@echo "  arduino-setup  Run Arduino CLI bootstrapper"
	@echo "  stm32-setup    Validate STM32CubeCLT paths and bootstrap STM32CubeN6 firmware"
	@echo "  prepare-dataset  Prepare OxIOD dataset (override OXIOD_ZIP=/path/to/OxIOD.zip)"
	@echo "  clean          Remove Python cache/build artifacts"

install:
	pip install -e . --no-deps

test:
	pytest test/

integration-test:
	RUN_INTEGRATION_TESTS=1 pytest test/integration/

test-all:
	RUN_INTEGRATION_TESTS=1 pytest test/

start-gpu:
	$(PYTHON) src/nas_model_client.py $(ARGS)

start-hil:
	$(PYTHON) src/hil_server.py

smoke:
	$(PYTHON) src/nas_model_client.py --smoke-test 1 $(ARGS)

env-create:
	conda env create -f environment.yml -n $(ENV)

arduino-setup:
	bash ./setup_arduino.sh

stm32-setup:
	bash ./setup_stm32.sh

prepare-dataset:
	$(PYTHON) data/dataset_download_and_splits/oxiod/prepare_oxiod.py --zip-path $(OXIOD_ZIP)

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
