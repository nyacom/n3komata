ROOT := $(abspath .)
ENGINE := $(ROOT)/engine/llama.cpp-n3k
JOBS ?= 4
PROFILE := flashnext-q3-adaptive

.PHONY: help setup build run web doctor clean

help:
	@printf '%s\n' 'make setup   - create the patched llama.cpp source tree' \
	  'make build   - build llama-cli and llama-server with Vulkan' \
	  'make run     - start interactive Qwen3.8 Flash Next chat' \
	  'make web     - start the local Web UI' \
	  'make clean   - remove generated engine source, build, and UI venv'

setup:
	python3 engine/materialize.py

build: setup
	cmake -S "$(ENGINE)" -B "$(ENGINE)/build" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build "$(ENGINE)/build" --target llama-cli llama-server -j$(JOBS)

run:
	N3KOMATA_ROOT="$(ROOT)" python3 ui/n3k chat $(PROFILE)

web:
	N3KOMATA_ROOT="$(ROOT)" python3 ui/n3k web $(PROFILE) --host 0.0.0.0

doctor:
	N3KOMATA_ROOT="$(ROOT)" python3 ui/n3k doctor $(PROFILE)

clean:
	python3 clean.py
