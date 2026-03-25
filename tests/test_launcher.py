from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from paperchat_cli import launcher


class FakeProcess:
    def __init__(self, *, exit_code: int | None = None) -> None:
        self._exit_code = exit_code

    def poll(self) -> int | None:
        return self._exit_code


def test_bootstrap_backend_runs_sync_when_virtualenv_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = launcher.RepoPaths(
        root=tmp_path,
        backend=tmp_path / "backend",
        frontend=tmp_path / "frontend",
    )
    paths.backend.mkdir()
    paths.frontend.mkdir()
    calls: list[tuple[list[str], Path, str]] = []

    monkeypatch.setattr(
        launcher,
        "run_command",
        lambda command, *, cwd, description: calls.append((command, cwd, description)),
    )

    launcher.bootstrap_backend(paths=paths, uv="/usr/bin/uv")

    assert calls == [
        (["/usr/bin/uv", "sync", "--group", "dev"], paths.backend, "backend sync"),
    ]


def test_bootstrap_frontend_skips_install_when_node_modules_exists(tmp_path: Path) -> None:
    paths = launcher.RepoPaths(
        root=tmp_path,
        backend=tmp_path / "backend",
        frontend=tmp_path / "frontend",
    )
    paths.backend.mkdir()
    paths.frontend.mkdir()
    vite_bin = paths.frontend / "node_modules" / ".bin"
    vite_bin.mkdir(parents=True)
    (vite_bin / "vite").write_text("#!/usr/bin/env bash\n")

    assert launcher.bootstrap_frontend(paths=paths, pnpm=None) == str(vite_bin / "vite")


def test_launch_uses_configured_backend_port_for_probe_and_frontend_api_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = tmp_path / "backend"
    frontend = tmp_path / "frontend"
    backend.mkdir()
    frontend.mkdir()
    paths = launcher.RepoPaths(root=tmp_path, backend=backend, frontend=frontend)
    wait_urls: list[str] = []
    process_calls: list[tuple[list[str], Path, dict[str, str] | None]] = []

    monkeypatch.setenv("PAPERCHAT_PORT", "9812")
    monkeypatch.setattr(launcher, "resolve_repo_paths", lambda: paths)
    monkeypatch.setattr(launcher, "require_command", lambda *_args, **_kwargs: "/usr/bin/uv")
    monkeypatch.setattr(launcher.shutil, "which", lambda _name: "/usr/bin/pnpm")
    monkeypatch.setattr(
        launcher,
        "bootstrap_backend",
        lambda *, paths, uv: None,  # noqa: ARG005
    )
    monkeypatch.setattr(
        launcher,
        "bootstrap_frontend",
        lambda *, paths, pnpm: "/usr/bin/vite",  # noqa: ARG005
    )
    monkeypatch.setattr(
        launcher,
        "run_migrations",
        lambda *, paths, uv: None,  # noqa: ARG005
    )
    monkeypatch.setattr(
        launcher,
        "start_process",
        lambda command, *, cwd, env_overrides=None: (
            process_calls.append(
                (command, cwd, dict(env_overrides) if env_overrides is not None else None)
            )
            or FakeProcess(exit_code=None)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "wait_for_url",
        lambda url, process, *, label, timeout_seconds=30.0: wait_urls.append(url),  # noqa: ARG005
    )
    monkeypatch.setattr(launcher, "monitor_processes", lambda *processes: None)
    monkeypatch.setattr(launcher, "stop_process", lambda process: None)

    launcher.launch(no_open=True)

    assert wait_urls == [
        "http://127.0.0.1:9812/api/health",
        "http://127.0.0.1:5173",
    ]
    assert process_calls == [
        (["/usr/bin/uv", "run", "paperchat-backend"], backend, None),
        (
            ["/usr/bin/vite", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
            frontend,
            {"VITE_API_URL": "http://127.0.0.1:9812"},
        ),
    ]


def test_wait_for_url_raises_when_process_exits_early() -> None:
    process = FakeProcess(exit_code=1)

    with pytest.raises(launcher.LaunchError, match="backend exited before becoming ready"):
        launcher.wait_for_url("http://127.0.0.1:1", process, label="backend", timeout_seconds=0.1)


def test_wait_for_url_retries_until_probe_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_probe(_url: str) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("not ready")

    monkeypatch.setattr(launcher, "probe_url", fake_probe)
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)

    launcher.wait_for_url(
        "http://127.0.0.1:1234",
        FakeProcess(exit_code=None),
        label="frontend",
        timeout_seconds=1,
    )

    assert attempts["count"] == 3


def test_run_command_raises_for_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    completed = subprocess.CompletedProcess(args=["uv"], returncode=2)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(launcher.LaunchError, match="backend sync failed with exit code 2"):
        launcher.run_command(["uv"], cwd=tmp_path, description="backend sync")
