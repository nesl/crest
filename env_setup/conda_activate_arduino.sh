#!/usr/bin/env bash
# Adds project-local Arduino CLI paths when the Conda env activates.

export CREST_ROOT="__PROJECT_ROOT__"
BIN_PATH="$CREST_ROOT/tools/bin"

export ARDUINO_DIRECTORIES_DATA="$CREST_ROOT/tools/arduino-data"
export ARDUINO_DIRECTORIES_DOWNLOADS="$CREST_ROOT/tools/arduino-downloads"
export ARDUINO_DIRECTORIES_USER="$CREST_ROOT/tools/arduino-user"
export ARDUINO_CONFIG_FILE="$CREST_ROOT/tools/arduino-cli.yaml"

if [[ ":$PATH:" != *":$BIN_PATH:"* ]]; then
  export _CREST_PREV_PATH="$PATH"
  export PATH="$BIN_PATH:$PATH"
else
  unset _CREST_PREV_PATH
fi
