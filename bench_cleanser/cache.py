"""Disk-based cache for LLM responses."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from datetime import UTC, datetime


class ResponseCache:
    """A simple disk-based cache for LLM responses.

    Cached entries are stored as JSON files in a nested directory structure
    using the first four hex characters of the SHA-256 key to create two
    levels of subdirectories:  ``cache_dir / key[:2] / key[2:4] / key.json``.

    Writes are atomic (write-to-tempfile then rename) to avoid corrupted
    reads from concurrent processes.
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = pathlib.Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(system_prompt: str, user_prompt: str, model: str) -> str:
        """Return a SHA-256 hex digest for the given prompt/model tuple."""
        content = system_prompt + user_prompt + model
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _key_path(self, key: str) -> pathlib.Path:
        """Return the file path for *key*."""
        return self._cache_dir / key[:2] / key[2:4] / f"{key}.json"

    def has(self, key: str) -> bool:
        """Return ``True`` if *key* exists in the cache."""
        return self._key_path(key).exists()

    def get(self, key: str) -> str | None:
        """Retrieve the cached response for *key*, or ``None``."""
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def delete(self, key: str) -> bool:
        """Remove the cached entry for *key*.  Returns True if it existed."""
        path = self._key_path(key)
        if path.exists():
            try:
                path.unlink()
                return True
            except OSError:
                return False
        return False

    def put(self, key: str, response: str, model: str = "") -> None:
        """Store *response* under *key* using an atomic write.

        Parameters
        ----------
        key:
            SHA-256 hex string (as returned by :meth:`make_key`).
        response:
            The raw LLM response text to cache.
        model:
            Optional model identifier stored alongside the response.
        """
        path = self._key_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(
            {
                "key": key,
                "model": model,
                "response": response,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )

        # Atomic write: create a temp file in the same directory, write,
        # then rename.  On Windows ``os.replace`` is atomic for most
        # filesystems; on POSIX it is guaranteed atomic.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up the temp file on failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @property
    def size(self) -> int:
        """Return the total number of cached entries."""
        count = 0
        for _root, _dirs, files in os.walk(self._cache_dir):
            for f in files:
                if f.endswith(".json"):
                    count += 1
        return count
