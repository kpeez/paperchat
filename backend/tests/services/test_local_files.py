from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from paperchat.services import local_files


def test_pick_documents_uses_osascript_on_macos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    first_pdf.write_text("one")
    second_pdf.write_text("two")

    monkeypatch.setattr(local_files.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_files.shutil, "which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(
        local_files.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=f"{first_pdf}\n{second_pdf}\n",
            stderr="",
        ),
    )

    assert local_files.pick_documents() == (str(first_pdf.resolve()), str(second_pdf.resolve()))


def test_pick_documents_returns_empty_tuple_when_user_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_files.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_files.shutil, "which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(
        local_files.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="User canceled.",
        ),
    )

    assert local_files.pick_documents() == ()


def test_pick_documents_raises_when_no_linux_picker_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_files.platform, "system", lambda: "Linux")
    monkeypatch.setattr(local_files.shutil, "which", lambda name: None)

    with pytest.raises(local_files.LocalFilePickerUnavailableError, match="zenity or kdialog"):
        local_files.pick_documents()
