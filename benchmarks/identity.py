"""Stable public pseudonyms; secrets and hardware UUIDs never leave this module."""
from __future__ import annotations

import hashlib
import hmac
import os
import stat
from pathlib import Path

KEY_HELP = "Initialize a 32-byte key explicitly outside the repository; see README HMAC key setup. No key was generated."


class IdentityMismatch(RuntimeError):
    """A proven physical-device contradiction; no identifiers in the message."""

    def __init__(self):
        super().__init__("physical_gpu_identity_mismatch")


def default_key_path():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gpu-unlock-research" / "hmac.key"


def _read_key(path):
    # Do not raise errors with paths or contents. Reject symlinks at open time
    # and check opened descriptors, rather than trusting a pre-open stat.
    key_path = Path(path) if path is not None else default_key_path()
    repo = Path(__file__).resolve().parent.parent
    directory_fd = file_fd = None
    try:
        key_path = key_path.expanduser().absolute()
        if key_path.resolve().is_relative_to(repo):
            return None, "key_must_be_outside_repository"
        directory_fd = os.open(key_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory = os.fstat(directory_fd)
        if stat.S_IMODE(directory.st_mode) != 0o700 or directory.st_uid != os.getuid():
            return None, "key_directory_requires_owner_and_mode_700"
        file_fd = os.open(key_path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        info = os.fstat(file_fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            return None, "key_file_requires_regular_owned_single_link_mode_600"
        if info.st_size != 32:
            return None, "key_requires_exactly_32_bytes"
        key = os.read(file_fd, 33)
        if len(key) != 32:
            return None, "key_requires_exactly_32_bytes"
        return key, None
    except FileNotFoundError:
        return None, "key_missing"
    except Exception:
        return None, "key_unavailable_or_unsafe_path"
    finally:
        for descriptor in (file_fd, directory_fd):
            if descriptor is not None:
                os.close(descriptor)


def gpu_pseudonym(uuid_reader, key_path=None):
    try:
        key, reason = _read_key(key_path)
        if key is None:
            return {"value": None, "status": "unavailable", "reason": reason, "initialization_help": KEY_HELP}
        # Only the digest survives; never put the UUID or any exception text in
        # a diagnostic result, including when NVML/encoding fails.
        uuid = uuid_reader()
        if isinstance(uuid, str):
            uuid = uuid.encode("ascii")
        if not isinstance(uuid, bytes) or not uuid.startswith(b"GPU-") or len(uuid) < 8:
            raise ValueError
        digest = hmac.new(key, b"gpu-unlock-research/gpu-id/v1\0" + uuid, hashlib.sha256).hexdigest()
        return {"value": "hmac-sha256:" + digest, "status": "available", "algorithm": "HMAC-SHA256-v1"}
    except Exception:
        return {"value": None, "status": "unavailable", "reason": "uuid_or_hmac_unavailable"}
