import math
import sys
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .config import DATA, PROFILES
from .control import request

ROOT = Path(__file__).resolve().parent.parent
LAUNCH = str(ROOT / "launch")
LOGO = ROOT / "assets" / "voicify.svg"
THEMES = {
    "dark": ("#0c0909", "#1a1010", "#492322", "#f7edeb", "#c3a9a5", "#df655b", "#f29483", "#190707"),
    "light": ("#f4ebe7", "#fffaf6", "#ddc1b8", "#321916", "#735752", "#9f2925", "#ab392f", "#ffffff"),
}


def label(text, role=""):
    item = QLabel(text)
    item.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        item.setObjectName(role)
    return item


def panel(layout, title, note):
    card = QFrame()
    card.setObjectName("Card")
    content = QVBoxLayout(card)
    content.setContentsMargins(22, 19, 22, 19)
    content.setSpacing(10)
    content.addWidget(label(title, "CardTitle"))
    description = label(note, "Muted")
    description.setWordWrap(True)
    content.addWidget(description)
    layout.addWidget(card)
    return content


class SignalTrace(QWidget):
    def __init__(self):
        super().__init__()
        self.samples = deque([0.] * 60, maxlen=60)
        self.setMinimumHeight(76)
        self.setAccessibleName("Recent virtual microphone output levels")
        self.colors = THEMES["dark"]

    def push(self, value):
        self.samples.append(min(1, math.sqrt(max(0, value)) * 1.3))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mid = self.height() / 2
        step = self.width() / len(self.samples)
        painter.setPen(QPen(QColor(self.colors[2]), 1))
        painter.drawLine(0, int(mid), self.width(), int(mid))
        for index, value in enumerate(self.samples):
            height = 2 + value * (mid - 5)
            x = (index + .5) * step
            tint = QColor(self.colors[5] if value else self.colors[2])
            painter.setPen(QPen(tint, max(2., step * .45), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(int(x), int(mid - height), int(x), int(mid + height))


class ImportDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Add a voice")
        self.setMinimumWidth(520)
        layout = QFormLayout(self)
        self.name = QLineEdit()
        self.encoder = QLineEdit()
        self.synth = QLineEdit()
        self.rate = QComboBox()
        self.rate.addItems(["32000", "40000", "48000"])
        self.rate.setCurrentText("40000")
        self.speaker = QSpinBox()
        self.speaker.setRange(0, 65535)
        layout.addRow("Name", self.name)
        for title, field in (("Content encoder", self.encoder), ("RVC voice", self.synth)):
            row = QHBoxLayout()
            row.addWidget(field)
            browse = QPushButton("Browse")
            browse.clicked.connect(lambda _=False, target=field: self.choose(target))
            row.addWidget(browse)
            layout.addRow(title, row)
        layout.addRow("Voice sample rate", self.rate)
        layout.addRow("Speaker ID", self.speaker)
        self.trust = QCheckBox("I trust these standalone ONNX files and can use this voice")
        self.trust.toggled.connect(self.validate)
        layout.addRow(self.trust)
        self.submit = QPushButton("Validate and import")
        self.submit.clicked.connect(self.accept)
        self.submit.setEnabled(False)
        layout.addRow(self.submit)
        for field in (self.name, self.encoder, self.synth):
            field.textChanged.connect(self.validate)

    def choose(self, field):
        value, _ = QFileDialog.getOpenFileName(self, "Choose an ONNX graph", "", "ONNX graphs (*.onnx)")
        if value:
            field.setText(value)

    def validate(self):
        self.submit.setEnabled(self.trust.isChecked() and all(
            field.text().strip() for field in (self.name, self.encoder, self.synth)
        ))

    def arguments(self):
        return ["add-voice", self.name.text().strip(), self.encoder.text(), self.synth.text(),
                "--rate", self.rate.currentText(), "--speaker", str(self.speaker.value())]


class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voicify Studio")
        self.setMinimumSize(880, 700)
        self.resize(990, 744)
        self.setCentralWidget(QWidget())
        root = QVBoxLayout(self.centralWidget())
        root.setContentsMargins(30, 23, 30, 18)
        root.setSpacing(14)
        self.connected = False
        self.pending_until = 0
        self.theme_name = "dark"
        self.preview_job = None
        self.import_job = None
        self.notice = ""
        self.capture_controls = {}
        self.mode_buttons = {}
        self.values = {}
        self.build_header(root)
        self.build_signal(root)
        self.build_controls(root)
        self.build_footer(root)
        self.apply_theme()
        self.timer = QTimer(self)
        self.timer.setInterval(180)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def build_header(self, root):
        self.setWindowIcon(QIcon(str(LOGO)))
        header = QHBoxLayout()
        mark = QLabel()
        mark.setPixmap(self.windowIcon().pixmap(53, 53))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(53, 53)
        header.addWidget(mark)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addWidget(label("VOICIFY  /  STUDIO", "Eyebrow"))
        brand.addWidget(label("Voice room", "Title"))
        header.addLayout(brand)
        header.addStretch()
        self.theme = QPushButton()
        self.theme.clicked.connect(self.flip_theme)
        self.service = QPushButton("Start audio")
        self.service.setObjectName("Primary")
        self.service.clicked.connect(self.toggle_service)
        header.addWidget(self.theme)
        header.addWidget(self.service)
        root.addLayout(header)

    def build_signal(self, root):
        hero = QFrame()
        hero.setObjectName("Hero")
        body = QVBoxLayout(hero)
        body.setContentsMargins(22, 15, 22, 14)
        body.setSpacing(6)
        status = QHBoxLayout()
        status.addWidget(label("VIRTUAL MICROPHONE", "Eyebrow"))
        status.addStretch()
        self.badge = label("●  OFFLINE", "Badge")
        status.addWidget(self.badge)
        body.addLayout(status)
        body.addWidget(label("Voicify Virtual Microphone", "HeroTitle"))
        body.addWidget(label("Choose this as your input in Discord, OBS or a game.", "Muted"))
        self.trace = SignalTrace()
        body.addWidget(self.trace)
        row = QGridLayout()
        self.input_meter, self.output_meter = QProgressBar(), QProgressBar()
        for column, (title, meter) in enumerate((("IN", self.input_meter), ("OUT", self.output_meter))):
            meter.setTextVisible(False)
            meter.setRange(0, 100)
            meter.setFixedHeight(6)
            meter.setAccessibleName(f"{title} level")
            row.addWidget(label(title, "Eyebrow"), 0, column * 2)
            row.addWidget(meter, 0, column * 2 + 1)
            row.setColumnStretch(column * 2 + 1, 1)
        row.setHorizontalSpacing(12)
        body.addLayout(row)
        root.addWidget(hero)

    def build_controls(self, root):
        columns = QHBoxLayout()
        columns.setSpacing(14)
        left, right = QVBoxLayout(), QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        root.addLayout(columns, 1)
        signal = panel(left, "Signal", "Pick a physical mic and decide what apps hear.")
        self.device = self.combo(signal, "Microphone", "device")
        self.device.addItem("Automatic microphone", "")
        modes = QHBoxLayout()
        group = QButtonGroup(self)
        for value, title in (("mute", "Mute"), ("direct", "Direct"), ("convert", "Convert")):
            button = QPushButton(title)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, mode=value: self.change(mode=mode))
            group.addButton(button)
            modes.addWidget(button)
            self.mode_buttons[value] = button
        self.mode_buttons["mute"].setChecked(True)
        signal.addLayout(modes)
        gain = QFormLayout()
        gain.setVerticalSpacing(7)
        self.spin(gain, "Input gain", "input_gain", 0, 200, 100, "%")
        self.spin(gain, "Output gain", "output_gain", 0, 200, 100, "%")
        self.spin(gain, "Noise gate", "gate", -80, -20, -48, " dB")
        self.values["gate"].setSpecialValueText("Off")
        signal.addLayout(gain)
        self.listen = QPushButton("Record preview")
        self.listen.clicked.connect(self.preview)
        self.listen.setToolTip("Record four seconds, then play the result through your headphones.")
        self.cleanup = QCheckBox("Echo cancellation")
        self.cleanup.toggled.connect(lambda enabled: self.change(cleanup=enabled))
        self.noise_suppression = QCheckBox("Extra noise filter")
        self.noise_suppression.setToolTip("Reduces background noise more strongly and may soften quiet speech.")
        self.noise_suppression.toggled.connect(lambda enabled: self.change(noise_suppression=enabled))
        cleanup_row = QHBoxLayout()
        cleanup_row.addWidget(self.cleanup)
        cleanup_row.addWidget(self.noise_suppression)
        cleanup_row.addWidget(self.listen, 1)
        signal.addLayout(cleanup_row)
        signal.addStretch()
        voice = panel(right, "Voice", "Choose an RVC model and tune how it runs.")
        self.voice = self.combo(voice, "Voice model", "model")
        self.voice.addItem("Select a voice model", "")
        actions = QHBoxLayout()
        add = QPushButton("Import ONNX")
        add.clicked.connect(self.import_voice)
        self.demo = QPushButton("Add demo voices")
        self.demo.clicked.connect(lambda: self.run_import(["demo-voices"]))
        actions.addWidget(add)
        actions.addWidget(self.demo)
        voice.addLayout(actions)
        tune = QFormLayout()
        tune.setVerticalSpacing(7)
        self.combo(tune, "Compute", "backend", ("Auto", "CPU", "CUDA"))
        self.combo(tune, "Buffer", "profile", tuple(PROFILES["CPU"])).setCurrentText("Balanced")
        self.spin(tune, "Pitch", "pitch", -24, 24, 0, " st")
        voice.addLayout(tune)
        self.credit = label("", "Muted")
        self.credit.setWordWrap(True)
        voice.addWidget(self.credit)
        voice.addStretch()

    def build_footer(self, root):
        row = QHBoxLayout()
        self.stats = label("ENGINE OFFLINE", "Eyebrow")
        row.addWidget(self.stats, 1)
        self.library = QPushButton("Voice folder")
        self.library.clicked.connect(self.open_library)
        row.addWidget(self.library)
        root.addLayout(row)
        self.message = label("Audio starts muted. Start the engine to begin.", "Muted")
        self.message.setWordWrap(True)
        root.addWidget(self.message)

    def combo(self, layout, title, key, choices=()):
        widget = QComboBox()
        widget.setAccessibleName(title)
        for option in choices:
            widget.addItem(option, option)
        widget.activated.connect(lambda _index: self.change(**{key: widget.currentData()}))
        self.capture_controls[key] = widget
        if isinstance(layout, QFormLayout):
            layout.addRow(title, widget)
        else:
            layout.addWidget(widget)
        return widget

    def spin(self, layout, title, key, low, high, default, suffix):
        widget = QSpinBox()
        widget.setRange(low, high)
        widget.setValue(default)
        widget.setSuffix(suffix)
        widget.editingFinished.connect(lambda: self.change(**{key: widget.value()}))
        layout.addRow(title, widget)
        self.values[key] = widget

    def toggle_service(self):
        if self.connected:
            if self.preview_job is not None:
                self.preview_job.kill()
            self.send("stop")
        else:
            self.pending_until = time.monotonic() + 5
            self.notice = "Starting the audio engine…"
            if not QProcess.startDetached(LAUNCH, ["start"], str(ROOT))[0]:
                self.notice = "Could not start the audio engine."
                self.pending_until = 0

    def send(self, action, **data):
        try:
            result = request(action, **data)
        except (OSError, ValueError) as exc:
            self.notice = str(exc)
            return None
        self.notice = ""
        return result

    def change(self, **settings):
        if self.preview_job is not None and any(key in settings for key in ("mode", "model", "cleanup", "noise_suppression")):
            self.preview_job.kill()
        if self.connected and self.send("update", settings=settings) is None:
            self.refresh()

    def preview(self):
        if self.preview_job is not None:
            self.preview_job.kill()
            return
        process = QProcess(self)
        self.preview_job = process
        self.listen.setText("Cancel preview")
        self.notice = "Recording for 4 seconds. Speak now."
        process.readyReadStandardOutput.connect(self.preview_playing)
        process.finished.connect(self.preview_finished)
        process.errorOccurred.connect(self.preview_error)
        process.start(LAUNCH, ["preview"])

    def preview_playing(self):
        self.preview_job.readAllStandardOutput()
        self.notice = "Playing your recorded preview…"

    def preview_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.preview_finished(1, QProcess.ExitStatus.NormalExit)

    def preview_finished(self, code, status):
        process = self.preview_job
        self.preview_job = None
        self.listen.setText("Record preview")
        error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        if status == QProcess.ExitStatus.CrashExit:
            self.notice = "Preview stopped."
        elif code:
            self.notice = error or "Could not start the preview."
        else:
            self.notice = "Preview finished."
        process.deleteLater()

    def refresh(self):
        try:
            state = request("status")
        except (OSError, ValueError):
            self.connected = False
            if self.pending_until and time.monotonic() >= self.pending_until:
                self.notice = "Audio engine did not start. Open its log for details."
                self.pending_until = 0
            self.render_offline()
            return
        self.connected = True
        if self.pending_until:
            self.notice = ""
        self.pending_until = 0
        self.service.setEnabled(True)
        self.service.setText("Stop audio")
        self.render_state(state)

    def render_offline(self):
        self.service.setText("Starting…" if self.pending_until else "Start audio")
        self.service.setEnabled(not self.pending_until)
        self.badge.setText("●  OFFLINE")
        self.stats.setText("ENGINE OFFLINE")
        self.trace.push(0)
        self.input_meter.setValue(0)
        self.output_meter.setValue(0)
        for widget in (*self.capture_controls.values(), *self.values.values(), *self.mode_buttons.values(), self.cleanup, self.noise_suppression, self.listen):
            widget.setEnabled(False)
        self.message.setText(self.notice or "Audio starts muted. Start the engine to begin.")

    def render_state(self, state):
        settings = state["settings"]
        items = [("Automatic microphone", "")] + [(d["label"], d["name"]) for d in state["devices"]]
        if settings["device"] and settings["device"] not in {value for _, value in items}:
            items.append(("Selected mic unavailable", settings["device"]))
        fill(self.device, items, settings["device"])
        choices = [("Select a voice model", "")] + [(m["name"], m["id"]) for m in state["voices"]]
        fill(self.voice, choices, settings["model"])
        active = next((m for m in state["voices"] if m["id"] == settings["model"]), {})
        self.credit.setText(active.get("credit", ""))
        for key, widget in self.capture_controls.items():
            widget.setEnabled(True)
            if not widget.view().isVisible():
                widget.setCurrentIndex(max(0, widget.findData(settings[key])))
        for key, widget in self.values.items():
            widget.setEnabled(True)
            if not widget.hasFocus() and widget.value() != settings[key]:
                widget.setValue(settings[key])
        self.values["pitch"].setEnabled(active.get("pitch_supported", False))
        self.values["pitch"].setToolTip("Pitch is available with F0-guided RVC voices.")
        for mode, button in self.mode_buttons.items():
            button.setEnabled(mode != "convert" or bool(settings["model"]))
            button.setChecked(mode == settings["mode"])
        for checkbox, key, default in ((self.cleanup, "cleanup", True),
                                        (self.noise_suppression, "noise_suppression", False)):
            checkbox.blockSignals(True)
            checkbox.setChecked(settings.get(key, default))
            checkbox.blockSignals(False)
        self.cleanup.setEnabled(True)
        self.noise_suppression.setEnabled(self.cleanup.isChecked())
        self.listen.setEnabled(self.preview_job is not None or state["registered"] and
                               state["state"] in ("Direct", "Converting"))
        self.badge.setText(f"●  {state['state'].upper()}")
        self.trace.push(state["output_level"])
        for meter, key in ((self.input_meter, "input_level"), (self.output_meter, "output_level")):
            level = state[key]
            meter.setValue(round(max(0, min(100, (20 * math.log10(max(level, .00001)) + 60) * 100 / 60))))
        details = ["PIPEWIRE ONLINE" if state["registered"] else "PIPEWIRE WAITING"]
        if state.get("cleanup_active"):
            details.append("CLEAN MIC")
        if state["backend"]:
            details.append(state["backend"])
        if state["buffer_ms"] is not None:
            details.append(f"{state['buffer_ms']} ms buffer")
        if state["processing_ms"] is not None:
            details.append(f"{state['processing_ms']:.0f} ms inference")
            details.append(f"{state['drops']} dropped")
        gpu = state["gpu"]
        if gpu:
            details.append(f"GPU {gpu['utilization']}%  /  {gpu['used']}/{gpu['total']} MiB")
        self.stats.setText("   ·   ".join(details))
        self.message.setText(self.notice or state.get("fault") or state["message"]
                             or ("Quiet room still showing a signal? Try a noise gate of −48 dB." if settings["gate"] < -65
                                 else "Audio stays on this device. Closing this window leaves the engine running."))

    def import_voice(self):
        dialog = ImportDialog(self)
        if dialog.exec():
            self.run_import(dialog.arguments())

    def run_import(self, arguments):
        if self.import_job is not None:
            return
        self.notice = "Checking the voice graphs and running a short inference test…"
        process = QProcess(self)
        process.finished.connect(self.import_finished)
        process.errorOccurred.connect(self.import_error)
        self.import_job = process
        self.demo.setEnabled(False)
        process.start(LAUNCH, arguments)

    def import_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.notice = "Could not start voice import."
            self.import_job.deleteLater()
            self.import_job = None
            self.demo.setEnabled(True)

    def import_finished(self, code, _status):
        process = self.import_job
        self.import_job = None
        self.demo.setEnabled(True)
        if code == 0:
            identifiers = bytes(process.readAllStandardOutput()).decode().split()
            if len(identifiers) == 1 and self.connected:
                self.change(model=identifiers[0])
            self.notice = "Voices ready. Choose a model and press Convert."
        else:
            error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
            QMessageBox.warning(self, "Voice import failed", error[-1600:] or "The importer stopped.")
            self.notice = "Voice import failed."
        process.deleteLater()

    def closeEvent(self, event):
        if self.preview_job is not None:
            self.preview_job.kill()
            self.preview_job.waitForFinished(500)
        super().closeEvent(event)

    def open_library(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        folder = DATA / "voices"
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def flip_theme(self):
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.apply_theme()

    def apply_theme(self):
        bg, card, line, text, muted, accent, mint, ink = THEMES[self.theme_name]
        self.theme.setText("Light theme" if self.theme_name == "dark" else "Dark theme")
        self.trace.colors = THEMES[self.theme_name]
        QApplication.instance().setStyleSheet(f"""
            QWidget {{ background: {bg}; color: {text}; font-size: 13px; }}
            QFrame#Card, QFrame#Hero {{ background: {card}; border: 1px solid {line}; border-radius: 18px; }}
            QLabel, QCheckBox, SignalTrace {{ background: transparent; }}
            QLabel#Title {{ font-size: 32px; font-weight: 750; }}
            QLabel#HeroTitle {{ font-size: 26px; font-weight: 720; }}
            QLabel#CardTitle {{ font-size: 20px; font-weight: 700; }}
            QLabel#Eyebrow {{ color: {accent}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
            QLabel#Muted {{ color: {muted}; }}
            QLabel#Badge {{ color: {mint}; font-weight: 750; font-size: 11px; }}
            QPushButton, QComboBox, QSpinBox, QLineEdit {{
                min-height: 30px; padding: 0 10px; border-radius: 8px;
                background: {bg}; border: 1px solid {line};
            }}
            QPushButton:hover, QComboBox:hover, QSpinBox:hover {{ border-color: {accent}; }}
            QPushButton:focus, QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {accent}; }}
            QPushButton:checked, QPushButton#Primary {{ background: {accent}; color: {ink}; font-weight: 700; }}
            QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled, QCheckBox:disabled {{ color: {muted}; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid {muted}; }}
            QCheckBox::indicator:checked {{ background: {mint}; border-color: {mint}; }}
            QCheckBox::indicator:disabled {{ border-color: {line}; }}
            QProgressBar {{ background: {line}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: {mint}; border-radius: 3px; }}
        """)


def fill(widget, items, current):
    existing = [(widget.itemText(i), widget.itemData(i)) for i in range(widget.count())]
    if items != existing and not widget.view().isVisible():
        widget.clear()
        for title, value in items:
            widget.addItem(title, value)
    index = widget.findData(current)
    if index >= 0 and not widget.view().isVisible():
        widget.setCurrentIndex(index)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Voicify")
    app.setDesktopFileName("voicify")
    app.setStyle("Fusion")
    app.setFont(QFont("Sans Serif", 10))
    studio = Studio()
    studio.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
