from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path


class LocalFilePickerUnavailableError(RuntimeError):
    """Raised when no native file picker is available on the current machine."""


def pick_documents() -> tuple[str, ...]:
    system = platform.system()
    if system == "Darwin":
        return _pick_macos_documents()
    if system == "Windows":
        return _pick_windows_documents()
    if system == "Linux":
        return _pick_linux_documents()

    raise LocalFilePickerUnavailableError(f"Native file picking is not supported on {system}.")


def _pick_macos_documents() -> tuple[str, ...]:
    osascript = _require_command(
        "osascript",
        message="macOS file picking requires the built-in osascript command.",
    )
    result = subprocess.run(
        [
            osascript,
            "-e",
            'set chosenFiles to choose file with prompt "Choose PDFs for PaperChat" '
            'of type {"com.adobe.pdf"} with multiple selections allowed true',
            "-e",
            "repeat with chosenFile in chosenFiles",
            "-e",
            "POSIX path of chosenFile",
            "-e",
            "end repeat",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return _parse_picker_result(result, cancel_markers=("User canceled.", "-128"))


def _pick_windows_documents() -> tuple[str, ...]:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        raise LocalFilePickerUnavailableError(
            "Windows file picking requires PowerShell or pwsh to be installed."
        )

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-STA",
            "-Command",
            (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.OpenFileDialog; "
                "$dialog.Filter = 'PDF files (*.pdf)|*.pdf'; "
                "$dialog.Multiselect = $true; "
                "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
                "{ $dialog.FileNames }"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return _parse_picker_result(result, cancel_markers=())


def _pick_linux_documents() -> tuple[str, ...]:
    zenity = shutil.which("zenity")
    if zenity is not None:
        result = subprocess.run(
            [
                zenity,
                "--file-selection",
                "--multiple",
                "--separator=\n",
                "--file-filter=PDF files | *.pdf",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return _parse_picker_result(result, cancel_markers=())

    kdialog = shutil.which("kdialog")
    if kdialog is not None:
        result = subprocess.run(
            [
                kdialog,
                "--getopenfilename",
                str(Path.home()),
                "*.pdf|PDF files",
                "--multiple",
                "--separate-output",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return _parse_picker_result(result, cancel_markers=())

    raise LocalFilePickerUnavailableError(
        "Linux file picking requires zenity or kdialog. Use manual path entry instead."
    )


def _parse_picker_result(
    result: subprocess.CompletedProcess[str],
    *,
    cancel_markers: tuple[str, ...],
) -> tuple[str, ...]:
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        if any(marker in stderr for marker in cancel_markers):
            return ()
        if result.returncode == 1 and not stderr:
            return ()
        raise LocalFilePickerUnavailableError(stderr or "The native file picker failed.")

    return tuple(_normalize_picker_output(result.stdout))


def _normalize_picker_output(stdout: str | None) -> tuple[str, ...]:
    if not stdout:
        return ()
    paths: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        resolved = Path(line).expanduser().resolve()
        if resolved.is_file():
            paths.append(str(resolved))
    return tuple(paths)


def _require_command(name: str, *, message: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise LocalFilePickerUnavailableError(message)
    return command
