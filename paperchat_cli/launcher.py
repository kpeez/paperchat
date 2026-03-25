from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BACKEND_PORT = 9712
HOST = "127.0.0.1"
FRONTEND_PORT = 5173
FRONTEND_URL = f"http://{HOST}:{FRONTEND_PORT}"


class LaunchError(RuntimeError):
    """Raised when the PaperChat launcher cannot complete startup."""


@dataclass(frozen=True, slots=True)
class RepoPaths:
    root: Path
    backend: Path
    frontend: Path


def launch(*, no_open: bool) -> None:
    paths = resolve_repo_paths()
    backend_url = get_backend_url()
    uv = require_command(
        "uv",
        message="uv is required to launch PaperChat. Install it from https://docs.astral.sh/uv/.",
    )
    pnpm = shutil.which("pnpm")

    bootstrap_backend(paths=paths, uv=uv)
    vite_command = bootstrap_frontend(paths=paths, pnpm=pnpm)
    run_migrations(paths=paths, uv=uv)

    backend_process: subprocess.Popen[str] | None = None
    frontend_process: subprocess.Popen[str] | None = None
    try:
        backend_process = start_process(
            [uv, "run", "paperchat-backend"],
            cwd=paths.backend,
        )
        wait_for_url(f"{backend_url}/api/health", backend_process, label="backend")

        frontend_process = start_process(
            [vite_command, "--host", HOST, "--port", str(FRONTEND_PORT), "--strictPort"],
            cwd=paths.frontend,
            env_overrides={"VITE_API_URL": backend_url},
        )
        wait_for_url(FRONTEND_URL, frontend_process, label="frontend")

        if not no_open:
            webbrowser.open(FRONTEND_URL)

        print(f"PaperChat is ready at {FRONTEND_URL}")
        monitor_processes(backend_process, frontend_process)
    except KeyboardInterrupt:
        print("\nStopping PaperChat...")
    finally:
        stop_process(frontend_process)
        stop_process(backend_process)


def resolve_repo_paths() -> RepoPaths:
    root = Path(__file__).resolve().parents[1]
    backend = root / "backend"
    frontend = root / "frontend"
    return RepoPaths(root=root, backend=backend, frontend=frontend)


def get_backend_port() -> int:
    return int(os.environ.get("PAPERCHAT_PORT", str(DEFAULT_BACKEND_PORT)))


def get_backend_url() -> str:
    return f"http://{HOST}:{get_backend_port()}"


def require_command(name: str, *, message: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise LaunchError(message)
    return command


def bootstrap_backend(*, paths: RepoPaths, uv: str) -> None:
    if (paths.backend / ".venv").exists():
        return
    print("Bootstrapping backend environment...")
    run_command([uv, "sync", "--group", "dev"], cwd=paths.backend, description="backend sync")


def bootstrap_frontend(*, paths: RepoPaths, pnpm: str | None) -> str:
    if not (paths.frontend / "node_modules").exists():
        if pnpm is None:
            raise LaunchError(
                "pnpm is required to install the frontend. Install Node.js 20+ and pnpm first."
            )
        print("Bootstrapping frontend environment...")
        run_command(
            [pnpm, "install", "--frozen-lockfile"],
            cwd=paths.frontend,
            description="frontend install",
        )

    vite_command = local_vite_command(paths.frontend)
    if not vite_command.exists():
        raise LaunchError(
            "The frontend Vite executable was not found after install. Re-run pnpm install in "
            "frontend/."
        )
    return str(vite_command)


def run_migrations(*, paths: RepoPaths, uv: str) -> None:
    print("Applying backend migrations...")
    run_command(
        [uv, "run", "alembic", "upgrade", "head"],
        cwd=paths.backend,
        description="backend migrations",
    )


def run_command(command: list[str], *, cwd: Path, description: str) -> None:
    result = subprocess.run(command, cwd=cwd, check=False, env=child_environment())
    if result.returncode != 0:
        raise LaunchError(f"{description} failed with exit code {result.returncode}.")


def start_process(
    command: list[str],
    *,
    cwd: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> subprocess.Popen[str]:
    kwargs: dict[str, object] = {
        "cwd": cwd,
        "env": child_environment(env_overrides=env_overrides),
        "text": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def wait_for_url(
    url: str,
    process: subprocess.Popen[str],
    *,
    label: str,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise LaunchError(f"{label} exited before becoming ready (exit code {exit_code}).")
        try:
            probe_url(url)
            return
        except Exception as error:  # pragma: no cover - exercised in polling tests
            last_error = error
            time.sleep(0.25)

    if last_error is None:
        raise LaunchError(f"{label} did not become ready before timing out.")
    raise LaunchError(f"{label} did not become ready before timing out: {last_error}")


def probe_url(url: str) -> None:
    with urllib.request.urlopen(url, timeout=1) as response:
        if response.status >= 400:
            raise LaunchError(f"HTTP {response.status} from {url}")


def monitor_processes(*processes: subprocess.Popen[str]) -> None:
    while True:
        for process in processes:
            exit_code = process.poll()
            if exit_code is not None:
                raise LaunchError(
                    f"A PaperChat child process exited unexpectedly (exit code {exit_code})."
                )
        time.sleep(0.5)


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
        return
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        pass

    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            return
    except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
        pass

    try:
        process.kill()
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def child_environment(*, env_overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    if env_overrides:
        environment.update(env_overrides)
    return environment


def local_vite_command(frontend_dir: Path) -> Path:
    executable = "vite.cmd" if os.name == "nt" else "vite"
    return frontend_dir / "node_modules" / ".bin" / executable
