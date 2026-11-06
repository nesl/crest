# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Coordinate hardware access across experiment runner processes."""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path


class DeviceBusyError(RuntimeError):
    """Raised when another process owns the requested device."""


@contextmanager
def device_lock(device_id, lock_dir=None):
    """Acquire an exclusive, nonblocking interprocess device lock.

    Parameters
    ----------
    device_id : str
        Stable device identity shared by all runners accessing that device.
        Callers must use the same identity for aliases of one physical device;
        only surrounding whitespace is normalized here.
    lock_dir : str or pathlib.Path, optional
        Shared lock directory. Defaults to ``CREST_DEVICE_LOCK_DIR`` or a
        per-user directory under ``/tmp``, independent of subprocess TMPDIR.

    Yields
    ------
    pathlib.Path
        Persistent lock file. It must not be removed while runners use it.

    Raises
    ------
    DeviceBusyError
        Another process already holds this device lock.
    ValueError
        The device identity is empty.
    RuntimeError
        The operating system does not support POSIX advisory file locks.
    """
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Device locking requires POSIX flock support") from exc
    identity = str(device_id).strip() if device_id is not None else ""
    if not identity:
        raise ValueError("device_id must be a nonempty stable device identity")
    if lock_dir is None:
        lock_dir = os.environ.get("CREST_DEVICE_LOCK_DIR") or (
            Path("/tmp") / f"crest-device-locks-{os.getuid()}"
        )
    directory = Path(lock_dir).expanduser()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".lock")
    # Keep the inode persistent: unlinking permits a second lock on a new inode.
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DeviceBusyError(
                f"Device {identity!r} is busy; another runner holds {path}"
            ) from exc
        try:
            yield path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
