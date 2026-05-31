#!/usr/bin/env bash
# Cleans up Arduino CLI env variables when the Conda env deactivates.

CREST_ROOT=${CREST_ROOT:-"__PROJECT_ROOT__"}

if [[ -n "${_CREST_PREV_PATH:-}" ]]; then
  export PATH="$_CREST_PREV_PATH"
  unset _CREST_PREV_PATH
fi

unset CREST_ROOT
unset ARDUINO_DIRECTORIES_DATA
unset ARDUINO_DIRECTORIES_DOWNLOADS
unset ARDUINO_DIRECTORIES_USER
unset ARDUINO_CONFIG_FILE
