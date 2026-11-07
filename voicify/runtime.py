import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox, QProgressDialog

ROOT = Path(__file__).resolve().parent.parent


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Voicify")
    app.setDesktopFileName("voicify")
    app.setWindowIcon(QIcon(str(ROOT / "assets" / "voicify.svg")))
    backend, accepted = QInputDialog.getItem(
        None, "Set up Voicify", "Choose a voice engine. Setup downloads the runtime into your user account.",
        ["CPU", "NVIDIA CUDA"], 0, False,
    )
    if not accepted:
        return 0
    process = QProcess()
    progress = QProgressDialog("Downloading the voice engine…", "Cancel", 0, 0)
    progress.setWindowTitle("Set up Voicify")
    progress.setMinimumDuration(0)
    process.finished.connect(progress.accept)
    process.errorOccurred.connect(progress.reject)
    progress.canceled.connect(process.kill)
    process.start(str(ROOT / "setup"), ["cuda" if backend == "NVIDIA CUDA" else "cpu"])
    if progress.exec() != QProgressDialog.DialogCode.Accepted:
        process.kill()
        process.waitForFinished(1000)
        return 1
    if process.exitCode():
        error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        QMessageBox.critical(None, "Voicify setup failed", error[-1800:] or "The runtime could not be installed.")
        return 1
    os.execv(str(ROOT / "launch"), [str(ROOT / "launch")])


if __name__ == "__main__":
    sys.exit(main())
