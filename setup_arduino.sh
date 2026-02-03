#!/usr/bin/env bash
# Sets up a project-local Arduino CLI plus Conda activation hooks.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
echo "Project root: $PROJECT_ROOT"
BIN_DIR="$PROJECT_ROOT/tools/bin"
CONFIG_FILE="$PROJECT_ROOT/tools/arduino-cli.yaml"
LIB_SUBMODULE="$PROJECT_ROOT/tools/arduino-user/libraries/Arduino_TensorFlowLite"

mkdir -p "$BIN_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download Arduino CLI" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please 'conda activate' the target environment before running this script." >&2
  exit 1
fi

echo "Downloading Arduino CLI into $BIN_DIR ..."
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
  | BINDIR="$BIN_DIR" sh

mkdir -p "$PROJECT_ROOT/tools"

echo "Initializing Arduino CLI config at $CONFIG_FILE ..."
"$BIN_DIR/arduino-cli" config init --config-file "$CONFIG_FILE" --overwrite

python - <<'PY'
import os
import pathlib
import yaml

project_root = pathlib.Path(os.environ["PROJECT_ROOT"])
cfg_path = project_root / "tools" / "arduino-cli.yaml"
tools_dir = project_root / "tools"
cfg = {}
if cfg_path.exists():
    cfg = yaml.safe_load(cfg_path.read_text()) or {}

cfg["directories"] = {
    "data": str(tools_dir / "arduino-data"),
    "downloads": str(tools_dir / "arduino-downloads"),
    "user": str(tools_dir / "arduino-user"),
}

cfg_path.write_text(yaml.safe_dump(cfg))
PY

echo "Updating core index and installing arduino:mbed_nano ..."
"$BIN_DIR/arduino-cli" core update-index --config-file "$CONFIG_FILE"
"$BIN_DIR/arduino-cli" core install arduino:mbed_nano --config-file "$CONFIG_FILE"
echo "Installing required Arduino libraries ..."
"$BIN_DIR/arduino-cli" lib install "Adafruit INA228 Library" --config-file "$CONFIG_FILE"

echo "Copying Conda activation hooks into $CONDA_PREFIX ..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
escaped_root="${PROJECT_ROOT//\//\\/}"
sed "s#__PROJECT_ROOT__#$escaped_root#g" "$PROJECT_ROOT/env_setup/conda_activate_arduino.sh" \
  > "$CONDA_PREFIX/etc/conda/activate.d/arduino.sh"
sed "s#__PROJECT_ROOT__#$escaped_root#g" "$PROJECT_ROOT/env_setup/conda_deactivate_arduino.sh" \
  > "$CONDA_PREFIX/etc/conda/deactivate.d/arduino.sh"
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/arduino.sh" "$CONDA_PREFIX/etc/conda/deactivate.d/arduino.sh"

cat <<EOF
Arduino CLI bootstrap complete.
- Binary location: $BIN_DIR/arduino-cli
- Config file: $CONFIG_FILE
- Conda hooks installed under $CONDA_PREFIX/etc/conda
- Arduino library root: $PROJECT_ROOT/tools/arduino-user/libraries
  $(if [[ -d "$LIB_SUBMODULE" ]]; then echo "TensorFlow Lite Micro examples detected."; else echo "TensorFlow Lite Micro repo missing; run 'git submodule update --init --recursive' to fetch tools/arduino-user/libraries/Arduino_TensorFlowLite."; fi)
Remember to add tools directories to .gitignore to keep binaries out of Git.
EOF
