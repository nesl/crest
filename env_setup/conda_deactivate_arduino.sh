#!/usr/bin/env bash
# Cleans up Arduino CLI env variables when the Conda env deactivates.

TINYODOM_ROOT=${TINYODOM_ROOT:-"__PROJECT_ROOT__"}

if [[ -n "${_TINYODOM_PREV_PATH:-}" ]]; then
  export PATH="$_TINYODOM_PREV_PATH"
  unset _TINYODOM_PREV_PATH
fi

unset TINYODOM_ROOT
unset ARDUINO_DIRECTORIES_DATA
unset ARDUINO_DIRECTORIES_DOWNLOADS
unset ARDUINO_DIRECTORIES_USER
unset ARDUINO_CONFIG_FILE
