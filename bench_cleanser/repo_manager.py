"""Repository cloning and caching for code visitation.

Manages shallow git clones of SWE-bench source repos, cached on disk
so that repeated pipeline runs reuse existing checkouts.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import threading
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Per-repo lock to prevent concurrent clone of the same repo+commit.
_clone_locks: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()


def _get_lock(key: str) -> threading.Lock:
    with _lock_guard:
        if key not in _clone_locks:
            _clone_locks[key] = threading.Lock()
        return _clone_locks[key]


def _repo_slug(repo: str) -> str:
    """Convert ``owner/name`` to ``owner__name`` for filesystem use."""
    return repo.replace("/", "__")


class RepoManager:
    """Clone and cache git repositories for code visitation.

    Cloned repos are stored under ``cache_dir/<repo_slug>/<commit[:12]>/``.
    A marker file ``.clone_complete`` indicates that the clone succeeded and
    can be reused.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/repos",
        clone_timeout: int = 300,
    ) -> None:
        self._cache_dir = pathlib.Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = clone_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_repo_path(self, repo: str, base_commit: str) -> pathlib.Path | None:
        """Return the local path for *repo* at *base_commit*, cloning if needed.

        Returns ``None`` if cloning fails.
        """
        slug = _repo_slug(repo)
        dest = self._cache_dir / slug / base_commit[:12]
        marker = dest / ".clone_complete"

        if marker.exists():
            logger.debug("Reusing cached clone: %s", dest)
            return dest

        lock = _get_lock(f"{slug}/{base_commit[:12]}")
        with lock:
            # Double-check after acquiring lock
            if marker.exists():
                return dest

            return self._clone(repo, base_commit, dest, marker)

    def get_file(self, repo_path: pathlib.Path, file_path: str) -> str | None:
        """Read a file from a cloned repo.  Returns ``None`` if not found."""
        full = repo_path / file_path
        if not full.exists():
            return None
        try:
            return full.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", full, exc)
            return None

    def get_files_for_task(
        self,
        repo_path: pathlib.Path,
        file_paths: Sequence[str],
    ) -> dict[str, str]:
        """Read multiple files from a cloned repo.  Skips missing files."""
        result: dict[str, str] = {}
        for fp in file_paths:
            content = self.get_file(repo_path, fp)
            if content is not None:
                result[fp] = content
        return result

    # ------------------------------------------------------------------
    # Batch pre-clone (call before pipeline processing)
    # ------------------------------------------------------------------

    def pre_clone_repos(
        self,
        tasks: list,
    ) -> dict[str, pathlib.Path | None]:
        """Clone all unique repos needed by *tasks*.

        *tasks* may be a list of ``TaskRecord`` objects (with ``.repo`` and
        ``.base_commit`` attributes) or a list of ``(repo, base_commit)``
        tuples.

        Returns a mapping from ``repo/commit`` to local path (or ``None``).
        """
        unique: dict[str, tuple[str, str]] = {}
        for item in tasks:
            if isinstance(item, tuple):
                repo, commit = item
            else:
                repo, commit = item.repo, item.base_commit
            key = f"{repo}/{commit[:12]}"
            if key not in unique:
                unique[key] = (repo, commit)

        logger.info(
            "Pre-cloning %d unique repo checkouts for %d tasks",
            len(unique),
            len(tasks),
        )

        results: dict[str, pathlib.Path | None] = {}
        for key, (repo, commit) in unique.items():
            path = self.get_repo_path(repo, commit)
            results[key] = path
            if path is None:
                logger.warning("Failed to clone %s @ %s", repo, commit[:12])
            else:
                logger.info("Ready: %s @ %s -> %s", repo, commit[:12], path)

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clone(
        self,
        repo: str,
        base_commit: str,
        dest: pathlib.Path,
        marker: pathlib.Path,
    ) -> pathlib.Path | None:
        """Perform the actual shallow clone + checkout."""
        url = f"https://github.com/{repo}.git"
        dest.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: init + configure remote idempotently
            self._run_git(["git", "init"], cwd=dest)
            remotes = self._run_git(["git", "remote"], cwd=dest).stdout.split()
            if "origin" in remotes:
                self._run_git(["git", "remote", "set-url", "origin", url], cwd=dest)
            else:
                self._run_git(["git", "remote", "add", "origin", url], cwd=dest)

            # Step 2: fetch the specific commit (shallow)
            self._run_git(
                ["git", "fetch", "--depth=1", "origin", base_commit],
                cwd=dest,
                timeout=self._timeout,
            )

            # Step 3: checkout
            self._run_git(
                ["git", "checkout", "FETCH_HEAD"],
                cwd=dest,
            )

            # Mark success
            marker.write_text("ok", encoding="utf-8")
            logger.info("Cloned %s @ %s -> %s", repo, base_commit[:12], dest)
            return dest

        except (subprocess.SubprocessError, OSError) as exc:
            logger.error("Clone failed for %s @ %s: %s", repo, base_commit[:12], exc)
            try:
                if dest.exists():
                    shutil.rmtree(dest)
            except OSError as cleanup_exc:
                logger.warning("Failed to clean partial clone at %s: %s", dest, cleanup_exc)
            return None

    def _run_git(
        self,
        cmd: list[str],
        cwd: pathlib.Path,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command with logging."""
        effective_timeout = timeout or self._timeout
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"  # never prompt for credentials

        logger.debug("Running: %s (cwd=%s)", " ".join(cmd), cwd)
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=True,
            env=env,
        )
