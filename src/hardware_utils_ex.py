from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Optional, Literal, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import logging, absl.logging
tf.get_logger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
logger = logging.getLogger(__name__)

import subprocess
import re
import time
import tempfile

import serial  


def _probe_xxd() -> Optional[str]:
    """Return the resolved xxd path when available."""
    candidate = shutil.which("xxd")
    if not candidate:
        return None
    try:
        # Some xxd builds (e.g., macOS 14) exit with status 1 for -h, so use
        # -v which cleanly reports the version and returns 0 when executable.
        subprocess.run(
            [candidate, "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return candidate


def _xxd_supports_custom_names(xxd_path: Optional[str]) -> bool:
    """Detect whether the host xxd binary accepts the `-n` flag."""
    if not xxd_path:
        return False
    temp_file: Optional[tempfile.NamedTemporaryFile] = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"\x00")
        temp_file.flush()
        temp_file.close()
        subprocess.run(
            [xxd_path, "-i", "-n", "probe_symbol", temp_file.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if temp_file is not None:
            Path(temp_file.name).unlink(missing_ok=True)
    return True


# Resolve the Arduino CLI executable once so every subprocess call uses the same path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REPO_ARDUINO_CLI = _PROJECT_ROOT / "tools" / "bin" / "arduino-cli"
ARDUINO_CLI_BIN = os.environ.get("ARDUINO_CLI_BIN")
ARDUINO_CLI_CONFIG = str(_PROJECT_ROOT / "tools" / "arduino-cli.yaml")
if not ARDUINO_CLI_BIN:
    if _REPO_ARDUINO_CLI.exists():
        ARDUINO_CLI_BIN = str(_REPO_ARDUINO_CLI)
    else:
        # Fallback to PATH lookup so developer installations still work.
        ARDUINO_CLI_BIN = shutil.which("arduino-cli") or "arduino-cli"
print(f"Using Arduino CLI at: {ARDUINO_CLI_BIN}")


XXD_BIN = _probe_xxd()
if XXD_BIN and not _xxd_supports_custom_names(XXD_BIN):
    XXD_BIN = None
if not XXD_BIN:
    print("xxd not found on PATH or doesn't support names; convert_to_cpp_model will use Python fallback.")

# -----------------------------------------------------------------------------
# Device catalog
# -----------------------------------------------------------------------------
"""
Add your target hardware device properties here.
Ex: Arduino Nano 33 BLE Sense added to device_list.
    SRAM: 256 KB -> usable maxRAM ≈ 200000 B (~78%) after RTOS/BLE overhead.
    Flash: 1 MB -> usable maxFlash ≈ 800000 B (leaves room for firmware + TFLM).
    Arena: preallocated memory buffer for TensorFlow Lite Micro tensors;
           defines candidate model sizes (here up to ~200 KB for 256 KB SRAM).
"""
DEVICE_SPECS = {
    "NUCLEO_F746ZG": {
        "arena_sizes": np.array([10, 30, 50, 75, 100, 150, 175, 200, 250, 280, 280]),
        "max_ram": 300_000, # notation for ease of reading, when accessed will be 300000
        "max_flash": 800_000,
        "fqbn": None,  # Mbed CLI workflow in the original project
    },
    "NUCLEO_L476RG": {
        "arena_sizes": np.array([10, 25, 40, 70, 85, 100, 100]),
        "max_ram": 100_000,
        "max_flash": 800_000,
        "fqbn": None,
    },
    "NUCLEO_F446RE": {
        "arena_sizes": np.array([10, 25, 40, 70, 85, 100, 100]),
        "max_ram": 100_000,
        "max_flash": 400_000,
        "fqbn": None,
    },
    "ARCH_MAX": {
        "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 160, 170, 170]),
        "max_ram": 180_000,
        "max_flash": 400_000,
        "fqbn": None,
    },
    # Arduino Nano 33 BLE Sense: nRF52840 microcontroller (Arm Cortex-M4, 64 MHz),
    # 256 KB SRAM, 1 MB flash. Tuned for TinyODOM with BLE support.
    "ARDUINO_NANO_33_BLE_SENSE": {
        "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 160, 180, 200, 210]),
        "max_ram": 215_000,  # Leave headroom for the RTOS/BLE stack.
        "max_flash": 800_000,
        "fqbn": "arduino:mbed_nano:nano33ble", # "fully qualified board name" how arduino specifies boards
    },
    # Arduino Nano RP2040 Connect: RP2040 microcontroller (dual-core Arm Cortex-M0+, 133 MHz),
    # 264 KB SRAM, 16 MB external flash. Great for ML with more RAM than Nano 33 BLE Sense.
    "ARDUINO_NANO_RP2040_CONNECT": {
    "arena_sizes": np.array([10, 25, 40, 70, 95, 120, 140, 150, 160, 180, 200, 210, 220]),
    "max_ram": 225_000,  # ~85% of 264 KB, leaving headroom for dual-core/Wi-Fi/BLE stack.
    "max_flash": 15_000_000,  # ~94% of 16 MB, plenty for large ML models.
    "fqbn": "arduino:mbed_nano:nano_rp2040_connect",
    },
}

# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------
def convert_to_tflite_model(
    model: tf.keras.Model,
    training_data,
    quantization: bool = False,
    output_name: Union[str, Path] = "g_model.tflite",
) -> None:
    """
    Export a Keras model to a TensorFlow Lite flatbuffer.

    Parameters
    ----------
    model : tf.keras.Model
        Source Keras model to serialize.
    training_data : array-like
        Calibration samples used when `quantization` is enabled.
    quantization : bool, optional
        Whether to apply post-training int8 quantization.
    output_name : Union[str, Path], optional
        Destination filename for the flatbuffer.

    Returns
    -------
    None
    """
    output_path = Path(output_name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    if quantization:
        data = np.asarray(training_data, dtype=np.float32)
        if data.ndim < 2:
            raise ValueError("`training_data` must include a sample dimension.")

        max_examples = min(len(data), 100)

        def representative_dataset() -> Iterable[Sequence[tf.Tensor]]:
            for sample in data[:max_examples]:
                # Yield calibrated batches so the converter can determine proper scale/zero-point.
                yield [tf.convert_to_tensor(sample[np.newaxis, ...], tf.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    flatbuffer = converter.convert()
    # Persist the flatbuffer so the downstream conversion step can embed it.
    output_path.write_bytes(flatbuffer)

def convert_to_cpp_model(
        tflite_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    array_name: str = "g_model",
    source_name: str = "model.cc",
    header_name: str = "model.h",
) -> Tuple[Path, Path]:
    """
    Materialize a `.tflite` flatbuffer as C sources for TensorFlow Lite Micro.
    Uses XXD when available, otherwise falls back to a Python implementation.

    Parameters
    ----------
    tflite_path : Union[str, Path]
        Path to the serialized TensorFlow Lite flatbuffer.
    output_dir : Union[str, Path]
        Destination directory for the generated source and header files.
    array_name : str, optional
        Symbol to use for the embedded byte array.
    source_name : str, optional
        Filename for the generated translation unit.
    header_name : str, optional
        Filename for the generated header.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Absolute paths to the generated `model.cc` and `model.h` files.
    """
    if XXD_BIN:
        return _convert_to_cpp_model_xxd(
            tflite_path,
            output_dir,
            array_name=array_name,
            source_name=source_name,
            header_name=header_name,
        )
    else:
        return _convert_to_cpp_model_python(
            tflite_path,
            output_dir,
            array_name=array_name,
            source_name=source_name,
            header_name=header_name,
        )


def _convert_to_cpp_model_python(
    tflite_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    array_name: str = "g_model",
    source_name: str = "model.cc",
    header_name: str = "model.h",
) -> Tuple[Path, Path]:
    """
    Materialize a `.tflite` flatbuffer as C sources for TensorFlow Lite Micro.
    Acts as a fallback when `xxd -i` is missing or the host version does not
    implement the `-n` flag (macOS 14). This guarantees the build tooling works
    on any developer machine but comes with two tradeoffs: conversion speed is
    slower than piping through `xxd`, and the emitted layout may differ slightly
    from the canonical TF Micro examples (e.g., indentation/line width).

    Parameters
    ----------
    tflite_path : Union[str, Path]
        Path to the serialized TensorFlow Lite flatbuffer.
    output_dir : Union[str, Path]
        Destination directory for the generated source and header files.
    array_name : str, optional
        Symbol to use for the embedded byte array.
    source_name : str, optional
        Filename for the generated translation unit.
    header_name : str, optional
        Filename for the generated header.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Absolute paths to the generated `model.cc` and `model.h` files.
    """
    tflite_path = Path(tflite_path)
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bytes = tflite_path.read_bytes()
    model_len = len(model_bytes)

    bytes_per_line = 12
    hex_lines = []
    # Emit the flatbuffer as a hex array with predictable row width for readability.
    for index in range(0, model_len, bytes_per_line):
        chunk = model_bytes[index: index + bytes_per_line]
        hex_line = ", ".join(f"0x{value:02x}" for value in chunk)
        hex_lines.append(f"  {hex_line},")
    if hex_lines:
        hex_lines[-1] = hex_lines[-1].rstrip(",")  # remove trailing comma on final line

    body = "\n".join(hex_lines)
    source = (
        f'#include "{header_name}"\n\n'
        f"alignas(8) const unsigned char {array_name}[] = {{\n"
        f"{body}\n"
        "};\n"
        f"const int {array_name}_len = {model_len};\n"
    )

    header = [
        "#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        "#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        f"extern const unsigned char {array_name}[];\n",
        f"extern const int {array_name}_len;\n",
        "#endif\n",
    ]

    source_path = output_dir / source_name
    header_path = output_dir / header_name
    source_path.write_text(source)
    header_path.write_text("".join(header))
    return source_path.resolve(), header_path.resolve()

def _convert_to_cpp_model_xxd(
    tflite_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    array_name: str = "g_model",
    source_name: str = "model.cc",
    header_name: str = "model.h",
) -> Tuple[Path, Path]:
    """
    Generate C sources by delegating to the `xxd -i` command-line tool.

    Parameters
    ----------
    tflite_path : Union[str, Path]
        Path to the `.tflite` model.
    output_dir : Union[str, Path]
        Destination directory for `model.cc`/`model.h`.
    array_name : str, optional
        Symbol to use for the generated array.
    source_name : str, optional
        Output translation unit filename.
    header_name : str, optional
        Output header filename.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Absolute paths to the generated source and header files.
    """
    # Use a temporary file so we can post-process the xxd output before copying it into the sketch.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(tflite_path)
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".cc", delete=False
    ) as temp_source:
        subprocess.run(
            ["xxd", "-i", "-n", array_name, str(model_path)],
            stdout=temp_source,
            check=True,
        )
        temp_source_path = Path(temp_source.name)

    lines = temp_source_path.read_text().splitlines(True)
    lines.insert(0, f'#include "{header_name}"\n')
    lines = [w.replace("unsigned int", "const int") for w in lines]
    lines = [w.replace("unsigned char", "alignas(8) const unsigned char") for w in lines]

    source_path = output_dir / source_name
    header_path = output_dir / header_name
    source_path.write_text("".join(lines))

    header = [
        "#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        "#define TENSORFLOW_LITE_MICRO_EXAMPLES_HELLO_WORLD_MODEL_H_\n",
        f"extern const unsigned char {array_name}[];\n",
        f"extern const int {array_name}_len;\n",
        "#endif\n",
    ]
    header_path.write_text("".join(header))

    temp_source_path.unlink(missing_ok=True)
    return source_path.resolve(), header_path.resolve()


# -----------------------------------------------------------------------------
# Hardware metadata accessors
# -----------------------------------------------------------------------------
def return_hardware_specs(device_name: str) -> Tuple[int, int]:
    """
    Retrieve RAM and flash limits for a supported device.

    Parameters
    ----------
    device_name : str
        Identifier present in DEVICE_SPECS.

    Returns
    -------
    Tuple[int, int]
        Maximum RAM bytes and flash bytes allowed on the device.
    """
    try:
        spec = DEVICE_SPECS[device_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown device '{device_name}'. Supported devices: {list(DEVICE_SPECS)}"
        ) from exc
    return spec["max_ram"], spec["max_flash"]


def get_model_memory_usage(
    batch_size,
    model,
    dtype_bytes: Optional[float] = None,
    quantized: bool = False,
):
    """
    Estimate the memory usage of a Keras model in bytes.

    Parameters
    ----------
    batch_size : int
        Batch size for inference.
    model : tf.keras.Model
        The Keras model to analyze.
    dtype_bytes : float, optional
        Override the bytes consumed per scalar value.
    quantized : bool, optional
        Treat tensors as int8 when ``dtype_bytes`` is not supplied.

    Returns
    -------
    float
        Total estimated memory usage in bytes.
    """
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            # Recursively calculate memory for nested models.
            internal_model_mem_count += get_model_memory_usage(
                batch_size,
                layer,
                dtype_bytes=dtype_bytes,
                quantized=quantized,
            )
        single_layer_mem = 1
        out_shape = getattr(layer, 'output_shape', None)
        if out_shape is None:
            # Some layers (e.g., InputLayer in newer Keras) do not expose output_shape.
            continue
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            # Multiply dimensions to get total elements per layer output.
            single_layer_mem *= s
        # Accumulate total elements across all layers.
        shapes_mem_count += single_layer_mem

    # Count trainable parameters using TF 2.x backend.
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    # Count non-trainable parameters using TF 2.x backend.
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Determine byte size based on Keras float precision.
    if dtype_bytes is not None:
        number_size = float(dtype_bytes)
    else:
        if quantized:
            number_size = 1.0
        else:
            number_size = 4.0
            if K.floatx() == 'float16':
                number_size = 2.0
            if K.floatx() == 'float64':
                number_size = 8.0

    # Calculate total memory: activations + parameters.
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    bytes_size = (total_memory + internal_model_mem_count)
    return bytes_size


def arena_size_candidates(device_name: str) -> np.ndarray:
    """
    Return the tensor-arena sweep (in kilobytes) for a device.

    Parameters
    ----------
    device_name : str
        Identifier present in DEVICE_SPECS.

    Returns
    -------
    numpy.ndarray
        Candidate arena sizes expressed in KiB.
    """
    try:
        spec = DEVICE_SPECS[device_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown device '{device_name}'. Supported devices: {list(DEVICE_SPECS)}"
        ) from exc
    return spec["arena_sizes"]


# -----------------------------------------------------------------------------
# Hardware-in-the-loop preparation
# -----------------------------------------------------------------------------
HIL_ERROR_OK = 0
HIL_ERROR_COMPILE = 1            # Compile failure
HIL_ERROR_LATENCY = 2            # Latency capture timed out (try larger arena)
HIL_ERROR_UNDER_SIZED = 3        # Runtime reported insufficient tensor arena
HIL_ERROR_FLASH_OVERFLOW = 4     # Sketch exceeds MCU flash limits
HIL_ERROR_RAM_OVERFLOW = 5       # Linker reports RAM overflow; retry with smaller arena
HIL_ERROR_UPLOAD = 6             # Upload/port failure (board missing or busy)

HIL_MASTER_PENDING = 0
HIL_MASTER_SUCCESS = 1
HIL_MASTER_ARENA_EXHAUSTED = 2
HIL_MASTER_FATAL = 3
HIL_MASTER_FLASH_OVERFLOW = 4
HIL_MASTER_RAM_OVERFLOW = 5
HIL_MASTER_DEVICE_NOT_FOUND = 6

_HIL_ERROR_LABELS = {
    HIL_ERROR_OK: "HIL_ERROR_OK",
    HIL_ERROR_COMPILE: "HIL_ERROR_COMPILE",
    HIL_ERROR_LATENCY: "HIL_ERROR_LATENCY",
    HIL_ERROR_UNDER_SIZED: "HIL_ERROR_UNDER_SIZED",
    HIL_ERROR_FLASH_OVERFLOW: "HIL_ERROR_FLASH_OVERFLOW",
    HIL_ERROR_RAM_OVERFLOW: "HIL_ERROR_RAM_OVERFLOW",
    HIL_ERROR_UPLOAD: "HIL_ERROR_UPLOAD",
}

_HIL_MASTER_ERROR_LABELS = {
    HIL_MASTER_PENDING: "HIL_MASTER_PENDING",
    HIL_MASTER_SUCCESS: "HIL_MASTER_SUCCESS",
    HIL_MASTER_ARENA_EXHAUSTED: "HIL_MASTER_ARENA_EXHAUSTED",
    HIL_MASTER_FATAL: "HIL_MASTER_FATAL",
    HIL_MASTER_FLASH_OVERFLOW: "HIL_MASTER_FLASH_OVERFLOW",
    HIL_MASTER_RAM_OVERFLOW: "HIL_MASTER_RAM_OVERFLOW",
    HIL_MASTER_DEVICE_NOT_FOUND: "HIL_MASTER_DEVICE_NOT_FOUND",
}


def describe_error_code(code: int, *, prefer_master: bool = True) -> str:
    """Return the symbolic name for a HIL or master error code."""
    lookup_order = (
        (_HIL_MASTER_ERROR_LABELS, _HIL_ERROR_LABELS)
        if prefer_master
        else (_HIL_ERROR_LABELS, _HIL_MASTER_ERROR_LABELS)
    )
    for table in lookup_order:
        if code in table:
            return table[code]
    return f"UNKNOWN_ERROR_{code}"

FLASH_USAGE_RE = re.compile(
    r"Sketch uses (\d+) bytes.*?Maximum is (\d+)", re.IGNORECASE | re.DOTALL
)
RAM_USAGE_RE = re.compile(
    r"Global variables use (\d+) bytes.*?Maximum is (\d+)", re.IGNORECASE | re.DOTALL
)
FLASH_OVERFLOW_PATTERNS = [
    re.compile(r"section [`']?\.text[`']?\s+will not fit in region [`']?flash[`']?", re.IGNORECASE),
    re.compile(r"region [`']?flash[`']?\s+overflowed", re.IGNORECASE),
]
RAM_OVERFLOW_PATTERNS = [
    re.compile(r"region [`']?ram[`']?\s+overflowed", re.IGNORECASE),
    re.compile(r"region [`']?sram[`']?\s+overflowed", re.IGNORECASE),
    re.compile(r"cannot move location counter backwards", re.IGNORECASE),
]
ARENA_TOO_SMALL_PATTERNS = [
    re.compile(r"size is too small for all buffers", re.IGNORECASE),
    re.compile(r"failed\s+to\s+allocate", re.IGNORECASE),
    re.compile(r"buffer\s+missing", re.IGNORECASE),
    # re.compile(r"fault", re.IGNORECASE),  # TODO: Uncomment if "Fault" lines are indicative of arena size issues. TBD
]
MISSING_BYTES_RE = re.compile(r"missing:\s*(\d+)", re.IGNORECASE)
REQUESTED_BYTES_RE = re.compile(r"requested:\s*(\d+)", re.IGNORECASE)
RETRY_BACKOFF_SECONDS = 1.0

_retry_hint_bytes: Optional[int] = None


def _store_retry_hint_bytes(value: Optional[int]) -> None:
    """Cache arena retry guidance for the most recent HIL_spec call."""
    global _retry_hint_bytes
    _retry_hint_bytes = value


def _pop_retry_hint_bytes() -> Optional[int]:
    """Fetch and clear the most recent arena retry hint (in bytes)."""
    global _retry_hint_bytes
    value = _retry_hint_bytes
    _retry_hint_bytes = None
    return value


def _compute_retry_hint_bytes(
    current_arena_bytes: int, arena_error_line: Optional[str]
) -> Optional[int]:
    """
    Derive a suggested arena size (bytes) from the device log when available.

    The firmware emits lines like:
    "Failed to resize buffer. Requested: 167360, available 113848, missing: 53512"
    We use the "missing" field (preferred) or "requested" to jump to a
    sufficiently large candidate on the next attempt.
    """
    if not arena_error_line:
        return None
    missing_match = MISSING_BYTES_RE.search(arena_error_line)
    requested_match = REQUESTED_BYTES_RE.search(arena_error_line)
    target_bytes: Optional[int] = None

    if missing_match:
        missing = int(missing_match.group(1))
        if missing > 0:
            target_bytes = current_arena_bytes + missing
    elif requested_match:
        requested = int(requested_match.group(1))
        if requested > current_arena_bytes:
            target_bytes = requested

    if target_bytes is None or target_bytes <= current_arena_bytes:
        return None

    # Add a small cushion to avoid oscillating on an exact boundary.
    return target_bytes + 2048


def _classify_compile_failure(log_text: str) -> Optional[Literal["flash", "ram"]]:
    """
    Determine whether the Arduino CLI output indicates a flash or RAM overflow.

    Parameters
    ----------
    log_text : str
        Concatenated stdout/stderr from `arduino-cli compile`.

    Returns
    -------
    Optional[Literal["flash", "ram"]]
        "flash" when program storage overflowed, "ram" for RAM overflow, otherwise None.
    """
    normalized = log_text.lower()
    for pattern in FLASH_OVERFLOW_PATTERNS:
        if pattern.search(normalized):
            return "flash"
    for pattern in RAM_OVERFLOW_PATTERNS:
        if pattern.search(normalized):
            return "ram"
    return None


def _replace_define(text: str, name: str, value: str) -> str:
    """
    Replace a single `#define` directive within the sketch source.

    Parameters
    ----------
    text : str
        Sketch contents to mutate.
    name : str
        Macro symbol to replace.
    value : str
        Replacement literal inserted after the symbol.

    Returns
    -------
    str
        Updated sketch text with the new macro definition.
    """
    pattern = re.compile(rf"(#define\s+{re.escape(name)}\s+)([^\n]+)")
    if not pattern.search(text):
        raise ValueError(f"Unable to locate definition for {name}.")
    # Use a callable replacement to avoid octal escape confusion when the new value is numeric.
    return pattern.sub(lambda match: f"{match.group(1)}{value}", text, count=1)


def _patch_sketch_constants(
    sketch_path: Path, arena_kb: int, window_size: int, num_channels: int
) -> None:
    """
    Rewrite TinyODOM deployment constants inside the Arduino sketch.

    Parameters
    ----------
    sketch_path : pathlib.Path
        Directory containing the target `.ino` file.
    arena_kb : int
        Tensor arena size expressed in KiB.
    window_size : int
        Sliding window length used by the model.
    num_channels : int
        Number of sensor channels captured per window.

    Returns
    -------
    None
    """
    ino_files = sorted(sketch_path.glob('*.ino'))
    if not ino_files:
        raise FileNotFoundError(f"No .ino file found in {sketch_path}")
    ino_path = ino_files[0]
    text = ino_path.read_text()
    text = _replace_define(text, 'TINYODOM_WINDOW_SIZE', str(window_size))
    text = _replace_define(text, 'TINYODOM_NUM_CHANNELS', str(num_channels))
    text = _replace_define(text, 'TINYODOM_TENSOR_ARENA_BYTES', f'({arena_kb} * 1024)')
    ino_path.write_text(text)


def _parse_memory_from_compile(output: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract flash and RAM usage from Arduino CLI compile output.

    Parameters
    ----------
    output : str
        stdout emitted by `arduino-cli compile`.

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        Flash bytes and RAM bytes when parseable, otherwise None placeholders.
    """
    # print("compile output:", output)  # Debug print
    flash_match = FLASH_USAGE_RE.search(output)
    ram_match = RAM_USAGE_RE.search(output)
    flash_bytes = int(flash_match.group(1)) if flash_match else None
    
    ram_bytes = int(ram_match.group(1)) if ram_match else None
    return flash_bytes, ram_bytes

# regex patterns for parsing power metrics from serial log.
_FLOAT_CAPTURE = r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
POWER_FIELD_SPECS: Dict[str, Tuple[str, re.Pattern[str]]] = {
    "sequence": (
        "int",
        re.compile(r"^inference seq:\s*(?P<value>\d+)$", re.IGNORECASE),
    ),
    # This is the important one for energy measurements.
    "energy_mj_per_inference": (
        "float",
        re.compile(rf"^energy output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "avg_power_mw": (
        "float",
        re.compile(rf"^avg power output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "avg_current_ma": (
        "float",
        re.compile(rf"^avg current output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "bus_voltage_v": (
        "float",
        re.compile(rf"^bus voltage output.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
    "idle_power_mw": (
        "float",
        re.compile(rf"^idle power baseline.*?:\s*{_FLOAT_CAPTURE}$", re.IGNORECASE),
    ),
}

POWER_METRIC_DEFAULTS: Dict[str, float] = {
    "sequence": -1.0,
    "energy_mj_per_inference": -1.0,
    "avg_power_mw": -1.0,
    "avg_current_ma": -1.0,
    "bus_voltage_v": -1.0,
    "idle_power_mw": -1.0,
}


def _parse_power_metrics(lines: Sequence[str]) -> Optional[Dict[str, Optional[float]]]:
    """
    Extract structured telemetry from the firmware serial log.

    Parses power-related metrics from a sequence of serial log lines using
    predefined regular expression patterns. Each metric is extracted only once
    from the first matching line; subsequent matches for the same metric are
    ignored (first match wins).

    Parameters
    ----------
    lines : Sequence[str]
        A sequence of strings representing the serial log lines from the firmware.

    Returns
    -------
    Optional[Dict[str, Optional[float]]]
        A dictionary containing the parsed power metrics if any were found,
        otherwise None. The keys correspond to metric names (e.g., 'energy_mj_per_inference',
        'avg_power_mw'), and values are floats or None if parsing failed.
        Possible keys include: 'sequence', 'energy_mj_per_inference', 'avg_power_mw',
        'avg_current_ma', 'bus_voltage_v', 'idle_power_mw'.
    """
    candidates: Dict[str, Optional[float]] = {key: None for key in POWER_FIELD_SPECS}
    matched = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        for key, (dtype, pattern) in POWER_FIELD_SPECS.items():
            if candidates[key] is not None:
                continue
            match = pattern.match(line)
            if not match:
                continue
            value = match.group("value")
            try:
                if dtype == "int":
                    candidates[key] = float(int(value))
                else:
                    candidates[key] = float(value)
                matched = True
            except (TypeError, ValueError):
                candidates[key] = None
    return candidates if matched else None


def normalize_power_metrics(power_metrics: Optional[Dict[str, Optional[float]]]) -> Dict[str, float]:
    """Return power metrics with defaults so downstream code can rely on keys."""

    normalized = POWER_METRIC_DEFAULTS.copy()
    if not power_metrics:
        return normalized
    for key, value in power_metrics.items():
        if key not in normalized or value is None:
            continue
        normalized[key] = value
    return normalized


def _collect_latency_seconds(
    port: str, baud: int, timeout_s: float
) -> Tuple[Optional[float], Optional[str], List[str]]:
    """
    Read the first `timer output:` line produced by the firmware.

    Parameters
    ----------
    port : str
        Serial port identifier.
    baud : int
        Serial baud rate.
    timeout_s : float
        Maximum time in seconds to wait for output.

    Returns
    -------
    Tuple[Optional[float], Optional[str], List[str]]
        Latency value in seconds (when available), the first arena error
        message detected while reading the serial log, and the decoded serial
        lines that were observed during the read window (for debugging).
    """
    decoded_lines: List[str] = []
    try:
        with serial.Serial(port, baudrate=baud, timeout=1.0) as ser:  # type: ignore[arg-type]
            start_time = time.time()
            while time.time() - start_time < timeout_s:
                raw = ser.readline()
                if not raw:
                    continue
                try:
                    line = raw.decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                decoded_lines.append(line)
    except serial.SerialException as exc:  # type: ignore[attr-defined]
        raise RuntimeError(f'Failed to read serial port {port}: {exc}') from exc
    for line in decoded_lines:
        lower_line = line.lower()
        if lower_line.startswith('timer output:'):
            _, _, value = line.partition(':')
            try:
                return float(value.strip()), None, decoded_lines
            except ValueError:
                return None, None, decoded_lines
        if any(pattern.search(lower_line) for pattern in ARENA_TOO_SMALL_PATTERNS):
            return None, line, decoded_lines
    return None, None, decoded_lines


def HIL_spec(
    dirpath: Union[str, Path] = 'tinyodom_tcn/',
    chosen_device: str = 'ARDUINO_NANO_33_BLE_SENSE',
    arenaSizes: Optional[Sequence[int]] = None,
    idx: int = 0,
    window_size: int = 400,
    number_of_channels: int = 6,
    serial_port: Optional[str] = None,
    baud_rate: int = 115200, # potentially highest baud rate for BLE 33
    serial_timeout_s: float = 12.0,
    compile_only: bool = False,
) -> Tuple[int, int, float, int, int, Optional[Dict[str, Optional[float]]]]:
    """
    Compile, deploy, and optionally profile TinyODOM on Arduino-class hardware.
    When ``compile_only`` is True the function stops after compilation so the
    caller can reuse the RAM/flash measurements without requiring a physical
    board. This keeps the objective function agnostic to whether HIL is
    connected while still sourcing metrics from the toolchain.

    Parameters
    ----------
    dirpath : Union[str, Path], optional
        Arduino sketch directory containing the TinyODOM sources.
    chosen_device : str, optional
        Hardware identifier that maps into DEVICE_SPECS.
    arenaSizes : Sequence[int], optional
        Custom arena sweep in KiB; defaults to the catalog entry.
    idx : int, optional
        Position inside the arena sweep to test.
    window_size : int, optional
        Sliding window length supplied to the firmware.
    number_of_channels : int, optional
        Number of sensor channels per window.
    serial_port : str, optional
        Serial port used for upload and latency capture.
    baud_rate : int, optional
        Serial baud rate for latency capture.
    serial_timeout_s : float, optional
        Seconds to wait for the `timer output:` line.
    compile_only : bool, optional
        Skip upload/latency capture and return compile metrics only.

    Returns
    -------
    Tuple[int, int, float, int, int, Optional[Dict[str, Optional[float]]]]
        Tuple of (RAM bytes, flash bytes, latency seconds, arena bytes, error flag,
        optional power telemetry parsed from the serial log).
    """
    _store_retry_hint_bytes(None)  # clear any stale hint from a previous call
    if chosen_device not in DEVICE_SPECS:
        raise ValueError(
            f"Unsupported device '{chosen_device}'. Supported devices: {list(DEVICE_SPECS)}"
        )

    spec = DEVICE_SPECS[chosen_device]
    if spec['fqbn'] is None:
        raise RuntimeError(
            f"No Arduino FQBN defined for {chosen_device}. Use the legacy Mbed workflow."
        )

    # Resolve the sketch path up-front so all subsequent operations use absolute paths.
    sketch_path = Path(dirpath).resolve()
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch directory not found: {sketch_path}")

    # Mirror the original HIL sweep: choose a single arena candidate for this attempt.
    arena_sweep_list = list(arenaSizes) if arenaSizes is not None else list(arena_size_candidates(chosen_device))
    if not arena_sweep_list:
        raise ValueError(f"No arena sizes registered for {chosen_device}.")
    if idx < 0 or idx >= len(arena_sweep_list):
        raise IndexError(f"arenaSizes index {idx} out of range for device {chosen_device}.")
    arena_kb = int(arena_sweep_list[idx])
    arena_bytes = arena_kb * 1024

    _patch_sketch_constants(sketch_path, arena_kb, window_size, number_of_channels)
    power_metrics: Optional[Dict[str, Optional[float]]] = None

    # Compile the sketch; Arduino CLI mirrors `mbed compile` but for Arduino cores.
    # Cache Arduino build artifacts inside the sketch folder so repeated compiles
    # can reuse previously compiled objects instead of starting from scratch.
    # This build directory is not cleaned up automatically to allow for faster
    # iterative development; users can delete it manually if desired. It is git-
    # ignored by default.
    build_cache_root = sketch_path / ".arduino-build"
    build_dir = build_cache_root / spec["fqbn"].replace(":", "_")
    build_dir.mkdir(parents=True, exist_ok=True)
    compile_cmd = [
        ARDUINO_CLI_BIN,
        '--config-file',
        ARDUINO_CLI_CONFIG,
        'compile',
        '--fqbn',
        spec['fqbn'],
        '--build-path',
        str(build_dir),
        str(sketch_path),
    ]
    compile_proc = subprocess.run(
        compile_cmd, capture_output=True, text=True, check=False
    )
    compile_log = f"{compile_proc.stdout}\n{compile_proc.stderr}"
    flash_bytes, ram_bytes = _parse_memory_from_compile(compile_log)
    overflow_kind = _classify_compile_failure(compile_log)
    
    # Even if we don't go far enough to overflow the compile step, if the ram size is larger than the max, we should flag it as an overflow. 
    if ram_bytes is not None and ram_bytes > spec["max_ram"]:
        overflow_kind = "ram"
    
    # If we overflowed during compilation, return early with the appropriate error code.
    if overflow_kind is not None:
        error_code = (
            HIL_ERROR_FLASH_OVERFLOW
            if overflow_kind == "flash"
            else HIL_ERROR_RAM_OVERFLOW
        )
        return (
            ram_bytes if ram_bytes is not None else -1,
            flash_bytes if flash_bytes is not None else -1,
            -1.0,
            arena_bytes,
            error_code,
            power_metrics,
        )
    
    if compile_proc.returncode != 0: # compilation failure
        print(f"Compilation failed: {compile_proc.stderr}")
        return (
            ram_bytes if ram_bytes is not None else -1,
            flash_bytes if flash_bytes is not None else -1,
            -1.0,
            arena_bytes,
            HIL_ERROR_COMPILE,
            power_metrics,
        )

    logger.info(
        "Compile succeeded: RAM=%s bytes, Flash=%s bytes, Arena=%s bytes",
        ram_bytes if ram_bytes is not None else "unknown",
        flash_bytes if flash_bytes is not None else "unknown",
        arena_bytes)

    if compile_only:
        # Compile-only mode surfaces RAM/flash/arena values without flashing hardware.
        return (
            ram_bytes if ram_bytes is not None else -1,
            flash_bytes if flash_bytes is not None else -1,
            -1.0,
            arena_bytes,
            HIL_ERROR_OK,
            power_metrics,
        )

    if serial_port is None:
        raise ValueError('serial_port must be provided when compile_only is False.')

    # Upload the sketch so the board can execute one inference and emit latency.
    # Construct the command to upload the compiled Arduino sketch to the board.
    # - ARDUINO_CLI_BIN: Path to the Arduino CLI executable (env override or repo-local binary).
    # - 'upload': Subcommand to flash the sketch to the connected board.
    # - '-p': Flag specifying the serial port (e.g., '/dev/ttyACM0') where the board is connected.
    # - serial_port: Variable holding the port string, determined earlier (e.g., from board detection).
    # - '--fqbn': Flag for the Fully Qualified Board Name, which identifies the board type and core (e.g., 'arduino:mbed:nano33blesense').
    # - spec['fqbn']: Retrieves the FQBN from the device specification dictionary for the chosen board.
    # - str(sketch_path): Path to the sketch directory containing the compiled binary to upload.
    upload_cmd = [
        ARDUINO_CLI_BIN,
        '--config-file',
        ARDUINO_CLI_CONFIG,
        'upload',
        '-p',
        serial_port,
        '--fqbn',
        spec['fqbn'],
        '--build-path',
        str(build_dir),
        str(sketch_path),
    ]
    upload_proc = subprocess.run(upload_cmd, capture_output=True, text=True, check=False)
    if upload_proc.returncode != 0: # upload failure
        logger.warning("Upload failed: %s", upload_proc.stderr)
        return (
            ram_bytes if ram_bytes is not None else -1,
            flash_bytes if flash_bytes is not None else -1,
            -1.0,
            arena_bytes,
            HIL_ERROR_UPLOAD,
            power_metrics,
        )

    # Capture the first `timer output:` line, which matches the legacy parser expectations.
    latency_s, arena_error_line, serial_log = _collect_latency_seconds(
        serial_port, baud_rate, serial_timeout_s
    )
    logger.info(
        "Latency capture result: latency_s=%s, arena_error_line=%s",
        latency_s if latency_s is not None else "None",
        arena_error_line if arena_error_line is not None else "None",)
    power_metrics = _parse_power_metrics(serial_log)
    
    if latency_s is None:
        if serial_log:
            logger.warning("Serial capture (%d lines): %s", len(serial_log), serial_log)
        hint_bytes = _compute_retry_hint_bytes(arena_bytes, arena_error_line)
        _store_retry_hint_bytes(hint_bytes)
        err_flag = HIL_ERROR_UNDER_SIZED if arena_error_line else HIL_ERROR_LATENCY
        return (
            ram_bytes if ram_bytes is not None else -1,
            flash_bytes if flash_bytes is not None else -1,
            -1.0,
            arena_bytes,
            err_flag,
            power_metrics,
        )

    return (
        ram_bytes if ram_bytes is not None else -1,
        flash_bytes if flash_bytes is not None else -1,
        latency_s,
        arena_bytes,
        HIL_ERROR_OK,
        power_metrics,
    )


def HIL_controller(
    dirpath: Union[str, Path] = 'tinyodom_tcn/',
    chosen_device: str = 'ARDUINO_NANO_33_BLE_SENSE',
    window_size: int = 400,
    number_of_channels: int = 6,
    serial_port: Optional[str] = None,
    baud_rate: int = 115200,
    serial_timeout_s: float = 12.0,
    run_hil: bool = True,
) -> Tuple[
    int,
    int,
    float,
    int,
    int,
    Optional[Dict[str, Optional[float]]],
]:
    """
    Search for the smallest arena size that compiles and runs successfully.
    ``run_hil`` toggles whether uploads/latency capture occur; when False the
    controller enters compile-only mode so offline Optuna trials can still rely
    on compiler-derived RAM/flash numbers for scoring.

    Parameters
    ----------
    dirpath : Union[str, Path], optional
        Arduino sketch directory containing TinyODOM sources.
    chosen_device : str, optional
        Hardware identifier that maps into DEVICE_SPECS.
    window_size : int, optional
        Sliding window length supplied to the firmware.
    number_of_channels : int, optional
        Number of sensor channels per window.
    serial_port : str, optional
        Serial port used for upload and latency capture.
    baud_rate : int, optional
        Serial baud rate for latency capture.
    serial_timeout_s : float, optional
        Timeout when waiting for the `timer output:` line.
    compile_only : bool, optional
        Skip flashing/measurement and return compile metrics only.

    Returns
    -------
    Tuple[int, int, float, int, int, Optional[Dict[str, Optional[float]]]]
        Final RAM bytes, flash bytes, latency seconds, arena bytes, error code, and
        optional power telemetry captured from the winning firmware run.
    """
    arena_sweep_list = list(arena_size_candidates(chosen_device))
    finRAM = -1
    finFlash = -1
    finLatency = -1.0
    idealArenaBytes = -1
    masterError = HIL_MASTER_PENDING
    last_success_metrics: Optional[Tuple[int, int, float, int, Optional[Dict[str, Optional[float]]]]] = None

    low_idx = -1
    high_idx = len(arena_sweep_list)

    def _next_candidate(low: int, high: int, preferred: int) -> Optional[int]:
        lower_bound = low + 1
        upper_bound = high - 1
        if lower_bound > upper_bound:
            return None
        return max(lower_bound, min(upper_bound, preferred))

    next_idx = _next_candidate(low_idx, high_idx, (low_idx + high_idx) // 2)
    iteration_count = 0
    max_iterations = max(1, len(arena_sweep_list) * 3)
    tested_bounds: dict[int, Tuple[int, int]] = {}
    finPower_metrics: Optional[Dict[str, Optional[float]]] = None

    compile_only = not run_hil  # compile-only allows proxy runs to reuse compiler metrics

    while (
        masterError == HIL_MASTER_PENDING
        and low_idx + 1 < high_idx
        and next_idx is not None
    ):
        iteration_count += 1
        if iteration_count > max_iterations:
            logger.error(
                "HIL_controller exceeded max iterations (%d); aborting to avoid infinite loop.",
                max_iterations,
            )
            masterError = HIL_MASTER_FATAL
            break

        current_idx = next_idx
        bounds_signature = (low_idx, high_idx)
        previous_bounds = tested_bounds.get(current_idx)
        if previous_bounds == bounds_signature:
            logger.error(
                "HIL_controller detected repeated idx=%d without bracket shrink (low=%d high=%d); aborting sweep.",
                current_idx,
                low_idx,
                high_idx,
            )
            masterError = HIL_MASTER_FATAL
            break
        tested_bounds[current_idx] = bounds_signature

        (
            ram_bytes,
            flash_bytes,
            latency_s,
            arena_bytes,
            err_flag,
            power_metrics,
        ) = HIL_spec(
            dirpath=dirpath,
            chosen_device=chosen_device,
            arenaSizes=arena_sweep_list,
            idx=current_idx,
            window_size=window_size,
            number_of_channels=number_of_channels,
            serial_port=serial_port,
            baud_rate=baud_rate,
            serial_timeout_s=serial_timeout_s,
            compile_only=compile_only,
        )
        retry_hint_bytes = _pop_retry_hint_bytes()

        logger.info(
            "HIL_controller attempt idx=%d arena=%d KiB err_flag=%d",
            current_idx,
            arena_sweep_list[current_idx],
            err_flag,
        )

        if err_flag != HIL_ERROR_OK:
            logger.warning(
                "HIL_controller failure reason: %s (err_flag=%d, arena=%d KiB, ram=%d bytes, flash=%d bytes, latency=%.3f s)",
                describe_error_code(err_flag, prefer_master=False),
                err_flag,
                arena_sweep_list[current_idx],
                ram_bytes,
                flash_bytes,
                latency_s,
            )

        if err_flag == HIL_ERROR_OK:
            # Successful run: capture metrics and stop searching.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            last_success_metrics = (finRAM, finFlash, finLatency, idealArenaBytes, finPower_metrics)
            high_idx = current_idx
            candidate = current_idx - 1
            next_idx = _next_candidate(low_idx, high_idx, candidate)
            if next_idx is None:
                masterError = HIL_MASTER_SUCCESS
            continue
        elif err_flag in (HIL_ERROR_LATENCY, HIL_ERROR_UNDER_SIZED):
            # Arena too small; advance to the next candidate.
            low_idx = max(low_idx, current_idx)
            candidate = min(current_idx + 1, (low_idx + high_idx) // 2)
            if retry_hint_bytes is not None:
                target_idx = next(
                    (i for i, kb in enumerate(arena_sweep_list) if kb * 1024 >= retry_hint_bytes),
                    None,
                )
                if target_idx is not None and target_idx > current_idx:
                    logger.info(
                        "Arena retry hint suggests jumping to idx=%d size=%d KiB (target_bytes=%d)",
                        target_idx,
                        arena_sweep_list[target_idx],
                        retry_hint_bytes,
                    )
                    candidate = max(candidate, target_idx)
            next_idx = _next_candidate(low_idx, high_idx, candidate)
            if next_idx is None:
                masterError = HIL_MASTER_SUCCESS if last_success_metrics else HIL_MASTER_ARENA_EXHAUSTED
            else:
                time.sleep(RETRY_BACKOFF_SECONDS)
            continue
        elif err_flag == HIL_ERROR_RAM_OVERFLOW:
            if current_idx == 0 and last_success_metrics is None:
                # Already at the smallest arena; surface RAM overflow upstream.     
                finRAM = ram_bytes
                finFlash = flash_bytes
                finLatency = latency_s
                idealArenaBytes = arena_bytes
                finPower_metrics = power_metrics
                masterError = HIL_MASTER_RAM_OVERFLOW
                break
            high_idx = min(high_idx, current_idx)
            next_idx = _next_candidate(low_idx, high_idx, (low_idx + high_idx) // 2)
            if next_idx is None:
                masterError = HIL_MASTER_SUCCESS if last_success_metrics else HIL_MASTER_ARENA_EXHAUSTED
            continue
        elif err_flag == HIL_ERROR_FLASH_OVERFLOW:
            # Exceeds flash/RAM limits; signal Optuna to prune.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            masterError = HIL_MASTER_FLASH_OVERFLOW
        elif err_flag == HIL_ERROR_UPLOAD:
            # Upload failures usually mean the board disappeared; stop the sweep immediately.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            masterError = HIL_MASTER_DEVICE_NOT_FOUND
            break
        else:
            # Non-arena failure (e.g., flash overflow). Surface immediately.
            finRAM = ram_bytes
            finFlash = flash_bytes
            finLatency = latency_s
            idealArenaBytes = arena_bytes
            finPower_metrics = power_metrics
            masterError = HIL_MASTER_FATAL

    if masterError != HIL_MASTER_SUCCESS and last_success_metrics is not None:
        finRAM, finFlash, finLatency, idealArenaBytes, finPower_metrics = last_success_metrics

    if masterError == HIL_MASTER_PENDING:
        masterError = HIL_MASTER_SUCCESS if last_success_metrics else HIL_MASTER_ARENA_EXHAUSTED

    if masterError != HIL_MASTER_SUCCESS:
        logger.warning(
            (
                "HIL_controller exiting with master error %s (code=%d, arena_bytes=%d, "
                "ram_bytes=%d, flash_bytes=%d, latency=%.3f s, power_metrics=%s)"
            ),
            describe_error_code(masterError),
            masterError,
            idealArenaBytes,
            finRAM,
            finFlash,
            finLatency,
            finPower_metrics,
        )
    return finRAM, finFlash, finLatency, idealArenaBytes, masterError, finPower_metrics
