#!/usr/bin/env bash
# Adds project-local Arduino CLI paths when the Conda env activates.

export TINYODOM_ROOT="__PROJECT_ROOT__"
BIN_PATH="$TINYODOM_ROOT/tools/bin"

export ARDUINO_DIRECTORIES_DATA="$TINYODOM_ROOT/tools/arduino-data"
export ARDUINO_DIRECTORIES_DOWNLOADS="$TINYODOM_ROOT/tools/arduino-downloads"
export ARDUINO_DIRECTORIES_USER="$TINYODOM_ROOT/tools/arduino-user"
export ARDUINO_CONFIG_FILE="$TINYODOM_ROOT/tools/arduino-cli.yaml"

if [[ ":$PATH:" != *":$BIN_PATH:"* ]]; then
  export _TINYODOM_PREV_PATH="$PATH"
  export PATH="$BIN_PATH:$PATH"
else
  unset _TINYODOM_PREV_PATH
fi
