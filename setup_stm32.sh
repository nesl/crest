#!/usr/bin/env bash
# Validates STM32CubeCLT tools on PATH and bootstraps a repo-local STM32CubeN6 checkout.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
STM32_TOOLS_DIR="$TOOLS_DIR/stm32"
FIRMWARE_DIR="$STM32_TOOLS_DIR/STM32CubeN6"
FIRMWARE_REPO_URL="https://github.com/STMicroelectronics/STM32CubeN6.git"
PINNED_TAG="v1.3.0"
CLT_INSTALL_URL="https://www.st.com/en/development-tools/stm32cubeclt.html"
GDBSERVER_BIN=""
GDB_BIN=""
CUBEPROG_CLI_BIN=""
CUBEPROG_BIN_DIR=""

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd is required to run this script." >&2
    exit 1
  fi
}

print_clt_guidance_and_exit() {
  cat >&2 <<EOF
STM32CubeCLT must be installed separately before this repo can bootstrap STM32 support.
Install it from:
  $CLT_INSTALL_URL

After installation, ensure these commands are available on PATH before rerunning:
  ST-LINK_gdbserver
  arm-none-eabi-gdb
  STM32_Programmer_CLI
  arm-none-eabi-gcc
  arm-none-eabi-size
  arm-none-eabi-objdump
EOF
  exit 1
}

validate_clt_tools() {
  GDBSERVER_BIN="$(command -v ST-LINK_gdbserver || true)"
  GDB_BIN="$(command -v arm-none-eabi-gdb || true)"
  CUBEPROG_CLI_BIN="$(command -v STM32_Programmer_CLI || true)"
  local resolved_cubeprog_cli
  local gcc_bin size_bin objdump_bin
  gcc_bin="$(command -v arm-none-eabi-gcc || true)"
  size_bin="$(command -v arm-none-eabi-size || true)"
  objdump_bin="$(command -v arm-none-eabi-objdump || true)"

  if [[ -z "$GDBSERVER_BIN" || -z "$GDB_BIN" || -z "$CUBEPROG_CLI_BIN" || -z "$gcc_bin" || -z "$size_bin" || -z "$objdump_bin" ]]; then
    [[ -z "$GDBSERVER_BIN" ]] && echo "Missing command on PATH: ST-LINK_gdbserver" >&2
    [[ -z "$GDB_BIN" ]] && echo "Missing command on PATH: arm-none-eabi-gdb" >&2
    [[ -z "$CUBEPROG_CLI_BIN" ]] && echo "Missing command on PATH: STM32_Programmer_CLI" >&2
    [[ -z "$gcc_bin" ]] && echo "Missing command on PATH: arm-none-eabi-gcc" >&2
    [[ -z "$size_bin" ]] && echo "Missing command on PATH: arm-none-eabi-size" >&2
    [[ -z "$objdump_bin" ]] && echo "Missing command on PATH: arm-none-eabi-objdump" >&2
    print_clt_guidance_and_exit
  fi

  resolved_cubeprog_cli="$(readlink -f "$CUBEPROG_CLI_BIN" 2>/dev/null || printf '%s' "$CUBEPROG_CLI_BIN")"
  CUBEPROG_BIN_DIR="$(cd "$(dirname "$resolved_cubeprog_cli")" && pwd)"
}

ensure_clean_git_checkout() {
  local repo_dir="$1"
  if [[ ! -d "$repo_dir/.git" ]]; then
    echo "Existing firmware path is not a git checkout: $repo_dir" >&2
    exit 1
  fi

  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    echo "Firmware checkout has local modifications: $repo_dir" >&2
    echo "Clean or remove it before rerunning setup_stm32.sh." >&2
    exit 1
  fi
}

sync_firmware_repo() {
  mkdir -p "$STM32_TOOLS_DIR"

  if [[ ! -e "$FIRMWARE_DIR" ]]; then
    echo "Cloning STM32CubeN6 into $FIRMWARE_DIR ..."
    git clone --recursive "$FIRMWARE_REPO_URL" "$FIRMWARE_DIR"
  else
    ensure_clean_git_checkout "$FIRMWARE_DIR"
  fi

  echo "Fetching STM32CubeN6 tags ..."
  git -C "$FIRMWARE_DIR" fetch --tags origin

  if ! git -C "$FIRMWARE_DIR" rev-parse --verify --quiet "$PINNED_TAG" >/dev/null; then
    echo "Pinned STM32CubeN6 tag was not found: $PINNED_TAG" >&2
    exit 1
  fi

  local current_commit pinned_commit
  current_commit="$(git -C "$FIRMWARE_DIR" rev-parse HEAD)"
  pinned_commit="$(git -C "$FIRMWARE_DIR" rev-list -n 1 "$PINNED_TAG")"

  if [[ "$current_commit" != "$pinned_commit" ]]; then
    echo "Checking out STM32CubeN6 $PINNED_TAG ..."
    git -C "$FIRMWARE_DIR" -c advice.detachedHead=false checkout "$PINNED_TAG"
  else
    echo "STM32CubeN6 is already on $PINNED_TAG."
  fi

  echo "Updating STM32CubeN6 submodules ..."
  git -C "$FIRMWARE_DIR" submodule update --init --recursive
}

main() {
  echo "Project root: $PROJECT_ROOT"

  require_command git
  validate_clt_tools
  sync_firmware_repo

  local firmware_ref
  firmware_ref="$(git -C "$FIRMWARE_DIR" describe --tags --exact-match 2>/dev/null || git -C "$FIRMWARE_DIR" rev-parse --short HEAD)"

  cat <<EOF
STM32 bootstrap complete.
- ST-LINK_gdbserver: $GDBSERVER_BIN
- arm-none-eabi-gdb: $GDB_BIN
- STM32_Programmer_CLI: $CUBEPROG_CLI_BIN
- STM32CubeProgrammer bin: $CUBEPROG_BIN_DIR
- Firmware root: $FIRMWARE_DIR
- Firmware ref: $firmware_ref
EOF
}

main "$@"
