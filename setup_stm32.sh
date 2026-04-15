#!/usr/bin/env bash
# Validates STM32CubeCLT tools on PATH and bootstraps a repo-local STM32CubeN6 checkout.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
STM32_TOOLS_DIR="$TOOLS_DIR/stm32"
FIRMWARE_DIR="$STM32_TOOLS_DIR/STM32CubeN6"
CANONICAL_TEMPLATE_ROOT="$PROJECT_ROOT/sketches/stm32/tinyodom_tcn_stm32/FSBL"
OWNERSHIP_MANIFEST="$PROJECT_ROOT/sketches/stm32/tinyodom_tcn_stm32/fsbl_ownership_manifest.tsv"
FIRMWARE_REPO_URL="https://github.com/STMicroelectronics/STM32CubeN6.git"
PINNED_TAG="v1.3.0"
CLT_INSTALL_URL="https://www.st.com/en/development-tools/stm32cubeclt.html"
declare -a EXTRA_TEMPLATE_ROOTS=(
  "$PROJECT_ROOT/analysis_scripts/stm32_example_project/stm32_blink_example_project/FSBL"
  "$PROJECT_ROOT/analysis_scripts/stm32_example_project/stm32_toy_ai_project/FSBL"
  "$PROJECT_ROOT/analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL"
)
GDBSERVER_BIN=""
GDB_BIN=""
CUBEPROG_CLI_BIN=""
CUBEPROG_BIN_DIR=""
declare -a MANIFEST_PATHS=()
declare -A CATEGORY_BY_PATH=()
declare -A SOURCE_BY_PATH=()

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file is missing: $path" >&2
    exit 1
  fi
}

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

load_ownership_manifest() {
  local category=""
  local relative_path=""
  local source_path=""

  require_file "$OWNERSHIP_MANIFEST"

  while IFS=$'\t' read -r category relative_path source_path; do
    [[ -z "$category" || "${category:0:1}" == "#" ]] && continue

    case "$category" in
      vendor_copy|vendor_derived|tinyodom_owned|generated|build_recipe)
        ;;
      *)
        echo "Unknown STM32 ownership category '$category' in $OWNERSHIP_MANIFEST" >&2
        exit 1
        ;;
    esac

    if [[ -z "$relative_path" ]]; then
      echo "Ownership manifest entry is missing a relative path." >&2
      exit 1
    fi

    if [[ -n "${CATEGORY_BY_PATH[$relative_path]:-}" ]]; then
      echo "Duplicate STM32 ownership entry for '$relative_path'." >&2
      exit 1
    fi

    if [[ "$category" == "vendor_copy" ]]; then
      if [[ -z "$source_path" ]]; then
        echo "Vendor-copy entry '$relative_path' is missing its CubeN6 source path." >&2
        exit 1
      fi
    elif [[ -n "$source_path" ]]; then
      echo "Only vendor-copy entries may declare a CubeN6 source path: $relative_path" >&2
      exit 1
    fi

    MANIFEST_PATHS+=("$relative_path")
    CATEGORY_BY_PATH["$relative_path"]="$category"
    if [[ -n "$source_path" ]]; then
      SOURCE_BY_PATH["$relative_path"]="$source_path"
    fi
  done < "$OWNERSHIP_MANIFEST"

  if [[ "${#MANIFEST_PATHS[@]}" -eq 0 ]]; then
    echo "STM32 ownership manifest is empty: $OWNERSHIP_MANIFEST" >&2
    exit 1
  fi
}

validate_ownership_manifest() {
  local relative_path=""
  local category=""
  local source_path=""
  local canonical_path=""

  for relative_path in "${MANIFEST_PATHS[@]}"; do
    category="${CATEGORY_BY_PATH[$relative_path]}"
    canonical_path="$CANONICAL_TEMPLATE_ROOT/$relative_path"

    case "$category" in
      vendor_copy)
        source_path="$FIRMWARE_DIR/${SOURCE_BY_PATH[$relative_path]}"
        require_file "$source_path"
        ;;
      vendor_derived|tinyodom_owned|build_recipe)
        require_file "$canonical_path"
        ;;
      generated)
        :
        ;;
    esac
  done
}

validate_tracked_template_files() {
  local canonical_prefix=""
  local tracked_path=""
  local relative_path=""
  local category=""

  case "$CANONICAL_TEMPLATE_ROOT" in
    "$PROJECT_ROOT"/*)
      canonical_prefix="${CANONICAL_TEMPLATE_ROOT#"$PROJECT_ROOT"/}"
      ;;
    *)
      return 0
      ;;
  esac

  while IFS= read -r tracked_path; do
    [[ -f "$PROJECT_ROOT/$tracked_path" ]] || continue
    relative_path="${tracked_path#"$canonical_prefix"/}"
    category="${CATEGORY_BY_PATH[$relative_path]:-}"

    case "$category" in
      vendor_derived|tinyodom_owned|build_recipe)
        ;;
      vendor_copy|generated)
        echo "Tracked canonical STM32 file must not live in git as '$category': $tracked_path" >&2
        exit 1
        ;;
      *)
        echo "Tracked canonical STM32 file is missing from the ownership manifest: $tracked_path" >&2
        exit 1
        ;;
    esac
  done < <(git -C "$PROJECT_ROOT" ls-files -- "$canonical_prefix")
}

assemble_canonical_template() {
  local relative_path=""
  local category=""
  local source_path=""
  local destination_path=""
  local -a missing_generated=()
  local vendor_copy_count=0

  mkdir -p "$CANONICAL_TEMPLATE_ROOT"

  # Vendor-copy files are not repo-owned. Refresh them from CubeN6 each run so
  # the canonical template stays pinned to the checked-out firmware tag.
  for relative_path in "${MANIFEST_PATHS[@]}"; do
    category="${CATEGORY_BY_PATH[$relative_path]}"
    destination_path="$CANONICAL_TEMPLATE_ROOT/$relative_path"

    case "$category" in
      vendor_copy)
        source_path="$FIRMWARE_DIR/${SOURCE_BY_PATH[$relative_path]}"
        mkdir -p "$(dirname "$destination_path")"
        cp "$source_path" "$destination_path"
        vendor_copy_count=$((vendor_copy_count + 1))
        ;;
      generated)
        if [[ ! -f "$destination_path" ]]; then
          missing_generated+=("$relative_path")
        fi
        ;;
    esac
  done

  if [[ "${#missing_generated[@]}" -gt 0 ]]; then
    cat <<EOF
STM32 canonical template assembled, but generated ST Edge AI files are absent:
$(printf '  - %s\n' "${missing_generated[@]}")
These files are intentionally not tracked. They may remain locally if already
generated, or be restaged later by the backend/analysis flows before a build.
EOF
  fi

  echo "Canonical STM32 template refreshed at $CANONICAL_TEMPLATE_ROOT"
  echo "Vendor-copy files materialized: $vendor_copy_count"
}

refresh_example_templates() {
  local template_root=""
  local relative_path=""
  local category=""
  local source_path=""
  local destination_path=""
  local vendor_copy_count=0

  for template_root in "${EXTRA_TEMPLATE_ROOTS[@]}"; do
    mkdir -p "$template_root"
    vendor_copy_count=0

    for relative_path in "${MANIFEST_PATHS[@]}"; do
      category="${CATEGORY_BY_PATH[$relative_path]}"
      [[ "$category" != "vendor_copy" ]] && continue

      source_path="$FIRMWARE_DIR/${SOURCE_BY_PATH[$relative_path]}"
      destination_path="$template_root/$relative_path"
      mkdir -p "$(dirname "$destination_path")"
      cp "$source_path" "$destination_path"
      vendor_copy_count=$((vendor_copy_count + 1))
    done

    echo "Example STM32 template refreshed at $template_root"
    echo "Vendor-copy files materialized: $vendor_copy_count"
  done
}

main() {
  echo "Project root: $PROJECT_ROOT"

  require_command git
  validate_clt_tools
  sync_firmware_repo
  load_ownership_manifest
  validate_ownership_manifest
  validate_tracked_template_files
  assemble_canonical_template
  refresh_example_templates

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
- Canonical template: $CANONICAL_TEMPLATE_ROOT
- Ownership manifest: $OWNERSHIP_MANIFEST
EOF
}

main "$@"
