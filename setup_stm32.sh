#!/usr/bin/env bash
# Validates STM32CubeCLT tools on PATH and bootstraps a repo-local STM32CubeN6 checkout.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$PROJECT_ROOT/tools"
STM32_TOOLS_DIR="$TOOLS_DIR/stm32"
FIRMWARE_DIR="$STM32_TOOLS_DIR/STM32CubeN6"
CANONICAL_LRUN_TEMPLATE_ROOT="$PROJECT_ROOT/sketches/stm32/tinyodom_stm32_lrun"
LRUN_OWNERSHIP_MANIFEST="$CANONICAL_LRUN_TEMPLATE_ROOT/lrun_ownership_manifest.tsv"
LRUN_UPSTREAM_TEMPLATE_ROOT_REL="Projects/NUCLEO-N657X0-Q/Templates/Template_FSBL_LRUN"
FIRMWARE_REPO_URL="https://github.com/STMicroelectronics/STM32CubeN6.git"
PINNED_TAG="v1.3.0"
CLT_INSTALL_URL="https://www.st.com/en/development-tools/stm32cubeclt.html"
GDBSERVER_BIN=""
GDB_BIN=""
CUBEPROG_CLI_BIN=""
CUBEPROG_BIN_DIR=""
SIGNING_TOOL_BIN=""
declare -a LRUN_MANIFEST_PATHS=()
declare -A LRUN_CATEGORY_BY_PATH=()
declare -A LRUN_SOURCE_BY_PATH=()
declare -a LRUN_OVERLAY_PATHS=(
  "Appli/Inc/mx25um51245g_conf.h"
  "Appli/Inc/stm32n6xx_hal_conf.h"
  "Appli/Inc/stm32n6xx_nucleo_conf.h"
  "Appli/Inc/tinyodom_dut_runner.h"
  "Appli/Src/tinyodom_dut_runner.c"
  "Appli/Src/main.c"
  "Appli/Src/stm32n6xx_hal_msp.c"
  "Appli/Src/stm32n6xx_it.c"
  "STM32CubeIDE/Boot/STM32N657XX_AXISRAM2_fsbl.ld"
  "STM32CubeIDE/AppS/STM32N657XX_LRUN.ld"
  "STM32CubeIDE/Boot/Debug"
  "STM32CubeIDE/AppS/Debug"
  "lrun_ownership_manifest.tsv"
  "README.md"
)
declare -a LRUN_APPLI_VENDOR_SOURCES=(
  "mx25um51245g.c::Drivers/BSP/Components/mx25um51245g/mx25um51245g.c"
  "stm32n6xx_hal.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal.c"
  "stm32n6xx_hal_cortex.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cortex.c"
  "stm32n6xx_hal_dma.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma.c"
  "stm32n6xx_hal_dma_ex.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma_ex.c"
  "stm32n6xx_hal_exti.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_exti.c"
  "stm32n6xx_hal_gpio.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_gpio.c"
  "stm32n6xx_hal_pwr.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr.c"
  "stm32n6xx_hal_pwr_ex.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr_ex.c"
  "stm32n6xx_hal_rcc.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc.c"
  "stm32n6xx_hal_rcc_ex.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc_ex.c"
  "stm32n6xx_hal_rtc.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rtc.c"
  "stm32n6xx_hal_rtc_ex.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rtc_ex.c"
  "stm32n6xx_hal_uart.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_uart.c"
  "stm32n6xx_hal_uart_ex.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_uart_ex.c"
  "stm32n6xx_hal_xspi.c::Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_xspi.c"
  "stm32n6xx_nucleo.c::Drivers/BSP/STM32N6xx_Nucleo/stm32n6xx_nucleo.c"
  "stm32n6xx_nucleo_xspi.c::Drivers/BSP/STM32N6xx_Nucleo/stm32n6xx_nucleo_xspi.c"
)

is_lrun_overlay_path() {
  local relative_path="$1"
  local overlay_path=""

  for overlay_path in "${LRUN_OVERLAY_PATHS[@]}"; do
    if [[ "$relative_path" == "$overlay_path" || "$relative_path" == "$overlay_path"/* ]]; then
      return 0
    fi
  done
  return 1
}

vendor_copy_state_file_for_root() {
  local template_root="$1"
  local relative_root=""
  local sanitized_root=""

  if [[ "$template_root" == "$PROJECT_ROOT"/* ]]; then
    relative_root="${template_root#"$PROJECT_ROOT"/}"
    printf '%s/.setup_state/%s.vendor_copy_paths\n' "$STM32_TOOLS_DIR" "$relative_root"
    return 0
  fi

  sanitized_root="${template_root//\//_}"
  printf '%s/.setup_state/external/%s.vendor_copy_paths\n' "$STM32_TOOLS_DIR" "$sanitized_root"
}

prune_empty_template_dirs() {
  local template_root="$1"
  local target_path="$2"
  local parent_dir=""

  parent_dir="$(dirname "$target_path")"
  while [[ "$parent_dir" != "$template_root" && "$parent_dir" != "/" ]]; do
    rmdir "$parent_dir" 2>/dev/null || break
    parent_dir="$(dirname "$parent_dir")"
  done
}

prune_recorded_state_paths() {
  local template_root="$1"
  local state_file=""
  local relative_path=""
  local destination_path=""

  state_file="$(vendor_copy_state_file_for_root "$template_root")"
  [[ -f "$state_file" ]] || return 0

  while IFS= read -r relative_path; do
    [[ -n "$relative_path" ]] || continue
    if [[ "$template_root" == "$CANONICAL_LRUN_TEMPLATE_ROOT" ]] && is_lrun_overlay_path "$relative_path"; then
      continue
    fi
    destination_path="$template_root/$relative_path"
    rm -f "$destination_path"
    prune_empty_template_dirs "$template_root" "$destination_path"
  done < "$state_file"
}

prune_materialized_vendor_copy_files() {
  local template_root="$1"
  local state_file=""
  local relative_path=""
  local destination_path=""
  local current_category=""

  state_file="$(vendor_copy_state_file_for_root "$template_root")"
  [[ -f "$state_file" ]] || return 0

  while IFS= read -r relative_path; do
    [[ -n "$relative_path" ]] || continue
    current_category="${LRUN_CATEGORY_BY_PATH[$relative_path]:-}"
    if [[ -n "$current_category" && "$current_category" != "vendor_copy" ]]; then
      continue
    fi
    destination_path="$template_root/$relative_path"
    rm -f "$destination_path"
    prune_empty_template_dirs "$template_root" "$destination_path"
  done < "$state_file"
}

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
  STM32_SigningTool_CLI (or STM32TrustedPackageCreator_CLI)
EOF
  exit 1
}

validate_clt_tools() {
  GDBSERVER_BIN="$(command -v ST-LINK_gdbserver || true)"
  GDB_BIN="$(command -v arm-none-eabi-gdb || true)"
  CUBEPROG_CLI_BIN="$(command -v STM32_Programmer_CLI || true)"
  local resolved_cubeprog_cli
  local gcc_bin size_bin objdump_bin signing_bin signing_fallback_bin
  gcc_bin="$(command -v arm-none-eabi-gcc || true)"
  size_bin="$(command -v arm-none-eabi-size || true)"
  objdump_bin="$(command -v arm-none-eabi-objdump || true)"
  signing_bin="$(command -v STM32_SigningTool_CLI || true)"
  signing_fallback_bin="$(command -v STM32TrustedPackageCreator_CLI || true)"
  SIGNING_TOOL_BIN="${signing_bin:-$signing_fallback_bin}"

  if [[ -z "$GDBSERVER_BIN" || -z "$GDB_BIN" || -z "$CUBEPROG_CLI_BIN" || -z "$gcc_bin" || -z "$size_bin" || -z "$objdump_bin" || -z "$SIGNING_TOOL_BIN" ]]; then
    [[ -z "$GDBSERVER_BIN" ]] && echo "Missing command on PATH: ST-LINK_gdbserver" >&2
    [[ -z "$GDB_BIN" ]] && echo "Missing command on PATH: arm-none-eabi-gdb" >&2
    [[ -z "$CUBEPROG_CLI_BIN" ]] && echo "Missing command on PATH: STM32_Programmer_CLI" >&2
    [[ -z "$gcc_bin" ]] && echo "Missing command on PATH: arm-none-eabi-gcc" >&2
    [[ -z "$size_bin" ]] && echo "Missing command on PATH: arm-none-eabi-size" >&2
    [[ -z "$objdump_bin" ]] && echo "Missing command on PATH: arm-none-eabi-objdump" >&2
    [[ -z "$SIGNING_TOOL_BIN" ]] && echo "Missing command on PATH: STM32_SigningTool_CLI or STM32TrustedPackageCreator_CLI" >&2
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

load_lrun_ownership_manifest() {
  local category=""
  local relative_path=""
  local source_path=""

  require_file "$LRUN_OWNERSHIP_MANIFEST"

  while IFS=$'\t' read -r category relative_path source_path; do
    [[ -z "$category" || "${category:0:1}" == "#" ]] && continue

    case "$category" in
      vendor_copy|vendor_derived|tinyodom_owned|generated|build_recipe)
        ;;
      *)
        echo "Unknown STM32 LRUN ownership category '$category' in $LRUN_OWNERSHIP_MANIFEST" >&2
        exit 1
        ;;
    esac

    if [[ -z "$relative_path" ]]; then
      echo "STM32 LRUN ownership entry is missing a relative path." >&2
      exit 1
    fi

    if [[ -n "${LRUN_CATEGORY_BY_PATH[$relative_path]:-}" ]]; then
      echo "Duplicate STM32 LRUN ownership entry for '$relative_path'." >&2
      exit 1
    fi

    if [[ "$category" == "vendor_copy" ]]; then
      if [[ -z "$source_path" ]]; then
        echo "STM32 LRUN vendor-copy entry '$relative_path' is missing its CubeN6 source path." >&2
        exit 1
      fi
    elif [[ -n "$source_path" ]]; then
      echo "Only STM32 LRUN vendor-copy entries may declare a CubeN6 source path: $relative_path" >&2
      exit 1
    fi

    LRUN_MANIFEST_PATHS+=("$relative_path")
    LRUN_CATEGORY_BY_PATH["$relative_path"]="$category"
    if [[ -n "$source_path" ]]; then
      LRUN_SOURCE_BY_PATH["$relative_path"]="$source_path"
    fi
  done < "$LRUN_OWNERSHIP_MANIFEST"

  if [[ "${#LRUN_MANIFEST_PATHS[@]}" -eq 0 ]]; then
    echo "STM32 LRUN ownership manifest is empty: $LRUN_OWNERSHIP_MANIFEST" >&2
    exit 1
  fi
}

validate_lrun_overlay_manifest() {
  local canonical_path=""
  local source_candidate=""
  local canonical_prefix=""
  local tracked_path=""
  local tracked_relative=""
  local tracked_category=""
  local relative_path=""
  local category=""

  for relative_path in "${LRUN_MANIFEST_PATHS[@]}"; do
    category="${LRUN_CATEGORY_BY_PATH[$relative_path]}"
    canonical_path="$CANONICAL_LRUN_TEMPLATE_ROOT/$relative_path"
    case "$category" in
      vendor_copy)
        source_candidate="$FIRMWARE_DIR/${LRUN_SOURCE_BY_PATH[$relative_path]}"
        require_file "$source_candidate"
        ;;
      vendor_derived|tinyodom_owned|build_recipe)
        require_file "$canonical_path"
        ;;
      generated)
        :
        ;;
    esac
  done

  case "$CANONICAL_LRUN_TEMPLATE_ROOT" in
    "$PROJECT_ROOT"/*)
      canonical_prefix="${CANONICAL_LRUN_TEMPLATE_ROOT#"$PROJECT_ROOT"/}"
      ;;
    *)
      return 0
      ;;
  esac

  while IFS= read -r tracked_path; do
    [[ -f "$PROJECT_ROOT/$tracked_path" ]] || continue
    tracked_relative="${tracked_path#"$canonical_prefix"/}"
    tracked_category="${LRUN_CATEGORY_BY_PATH[$tracked_relative]:-}"

    case "$tracked_category" in
      vendor_derived|tinyodom_owned|build_recipe)
        ;;
      vendor_copy|generated)
        echo "Tracked canonical STM32 LRUN file must not live in git as '$tracked_category': $tracked_path" >&2
        exit 1
        ;;
      *)
        echo "Tracked canonical STM32 LRUN file is missing from the ownership manifest: $tracked_path" >&2
        exit 1
        ;;
    esac
  done < <(git -C "$PROJECT_ROOT" ls-files -- "$canonical_prefix")
}

validate_lrun_workspace_structure() {
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/FSBL/Inc/stm32_extmem_conf.h"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/mx25um51245g_conf.h"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/stm32n6xx_hal_conf.h"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/tinyodom_dut_runner.h"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/tinyodom_dut_runner.c"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/system_stm32n6xx_s.c"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Secure_nsclib/secure_nsc.h"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/STM32CubeIDE/Boot/Debug/makefile"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/STM32CubeIDE/AppS/Debug/makefile"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/STM32CubeIDE/Boot/Debug/Src/subdir.mk"
  require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/STM32CubeIDE/AppS/Debug/Src/subdir.mk"
  if ! grep -q "FSBL/Inc" "$CANONICAL_LRUN_TEMPLATE_ROOT/STM32CubeIDE/Boot/Debug/Src/subdir.mk"; then
    echo "STM32 LRUN Boot debug recipes no longer include FSBL/Inc." >&2
    exit 1
  fi
  if ! grep -q "Secure_nsclib" "$CANONICAL_LRUN_TEMPLATE_ROOT/STM32CubeIDE/AppS/Debug/Src/subdir.mk"; then
    echo "STM32 LRUN AppS debug recipes no longer include Secure_nsclib." >&2
    exit 1
  fi
}

sync_lrun_template() {
  local upstream_root="$FIRMWARE_DIR/$LRUN_UPSTREAM_TEMPLATE_ROOT_REL"
  local overlay_path=""

  require_command rsync
  if [[ ! -d "$upstream_root" ]]; then
    echo "Missing STM32 LRUN upstream template: $upstream_root" >&2
    exit 1
  fi
  mkdir -p "$CANONICAL_LRUN_TEMPLATE_ROOT"
  prune_recorded_state_paths "$CANONICAL_LRUN_TEMPLATE_ROOT"

  local -a rsync_args=(
    -a
    --delete
  )
  for overlay_path in "${LRUN_OVERLAY_PATHS[@]}"; do
    rsync_args+=(--exclude "$overlay_path")
  done

  rsync "${rsync_args[@]}" "$upstream_root/" "$CANONICAL_LRUN_TEMPLATE_ROOT/"

  rm -f \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/network.h" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/network_config.h" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/network_data.h" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/network_data_params.h" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/tinyodom_dut_phase_config.h" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/network.c" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/network_data.c" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/network_data_params.c" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/network_c_info.json" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/network_generate_report.txt" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/network_generate_report.json"

  echo "Canonical STM32 LRUN template refreshed at $CANONICAL_LRUN_TEMPLATE_ROOT"
}

sync_lrun_support_tree() {
  local mapping=""
  local dest_name=""
  local source_rel=""
  local source_path=""
  local dest_path=""

  mkdir -p "$CANONICAL_LRUN_TEMPLATE_ROOT/Drivers"
  mkdir -p "$CANONICAL_LRUN_TEMPLATE_ROOT/Middlewares/ST"
  mkdir -p "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc"
  mkdir -p "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src"

  rsync -a --delete "$FIRMWARE_DIR/Drivers/" "$CANONICAL_LRUN_TEMPLATE_ROOT/Drivers/"
  rsync -a --delete \
    "$FIRMWARE_DIR/Middlewares/ST/STM32_ExtMem_Manager/" \
    "$CANONICAL_LRUN_TEMPLATE_ROOT/Middlewares/ST/STM32_ExtMem_Manager/"

  for mapping in "${LRUN_APPLI_VENDOR_SOURCES[@]}"; do
    dest_name="${mapping%%::*}"
    source_rel="${mapping##*::}"
    source_path="$FIRMWARE_DIR/$source_rel"
    dest_path="$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/$dest_name"
    require_file "$source_path"
    cp "$source_path" "$dest_path"
  done
}

record_lrun_materialized_files() {
  local state_file=""
  local state_dir=""
  local file_path=""
  local relative_path=""
  local category=""

  state_file="$(vendor_copy_state_file_for_root "$CANONICAL_LRUN_TEMPLATE_ROOT")"
  state_dir="$(dirname "$state_file")"
  mkdir -p "$state_dir"
  : > "$state_file"

  while IFS= read -r file_path; do
    relative_path="${file_path#"$CANONICAL_LRUN_TEMPLATE_ROOT"/}"
    if is_lrun_overlay_path "$relative_path"; then
      continue
    fi
    category="${LRUN_CATEGORY_BY_PATH[$relative_path]:-}"
    case "$category" in
      tinyodom_owned|vendor_derived|build_recipe|generated)
        continue
        ;;
    esac
    printf '%s\n' "$relative_path" >> "$state_file"
  done < <(find "$CANONICAL_LRUN_TEMPLATE_ROOT" -type f | sort)
}

main() {
  echo "Project root: $PROJECT_ROOT"

  require_command git
  validate_clt_tools
  sync_firmware_repo
  load_lrun_ownership_manifest
  sync_lrun_template
  sync_lrun_support_tree
  record_lrun_materialized_files
  validate_lrun_overlay_manifest
  validate_lrun_workspace_structure

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
- Canonical LRUN template: $CANONICAL_LRUN_TEMPLATE_ROOT
- LRUN ownership manifest: $LRUN_OWNERSHIP_MANIFEST
- LRUN signing tool: $SIGNING_TOOL_BIN
EOF
}

main "$@"
