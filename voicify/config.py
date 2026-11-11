import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

CONFIG = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "voicify"
DATA = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share")) / "voicify"
PROFILES = {
    "CUDA": {"Quick": 80, "Balanced": 120, "Smooth": 180},
    "CPU": {"Quick": 180, "Balanced": 240, "Smooth": 360},
}
MODES = ("mute", "direct", "pitch", "convert")


def socket_path():
    return Path(os.environ["XDG_RUNTIME_DIR"]) / "voicify" / "control.sock"


def write_json(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Settings:
    device: str = ""
    model: str = ""
    backend: str = "Auto"
    profile: str = "Balanced"
    mode: str = "mute"
    pitch: int = 0
    input_gain: int = 100
    output_gain: int = 100
    gate: int = -48
    cleanup: bool = True
    noise_suppression: bool = False

    def update(self, changes):
        updated = replace(self, **changes)
        for name in ("device", "model"):
            value = getattr(updated, name)
            if type(value) is not str or len(value) > 255:
                raise ValueError(f"Invalid {name}")
        if updated.model and (len(updated.model) != 64 or set(updated.model) - set("0123456789abcdef")):
            raise ValueError("Invalid model ID")
        if updated.backend not in ("Auto", "CPU", "CUDA") or updated.profile not in PROFILES["CPU"]:
            raise ValueError("Invalid performance setting")
        if updated.mode not in MODES or (updated.mode == "convert" and not updated.model):
            raise ValueError("Choose a voice model before enabling conversion")
        if type(updated.cleanup) is not bool:
            raise ValueError("Cleanup must be on or off")
        if type(updated.noise_suppression) is not bool:
            raise ValueError("Noise suppression must be on or off")
        limits = {"pitch": (-24, 24), "input_gain": (0, 200), "output_gain": (0, 200), "gate": (-80, -20)}
        for name, (low, high) in limits.items():
            value = getattr(updated, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}")
        return updated

    @classmethod
    def load(cls):
        try:
            changes = json.loads((CONFIG / "settings.json").read_text())
            return cls().update(changes).update({"mode": "mute"})
        except FileNotFoundError:
            return cls()

    def save(self):
        write_json(CONFIG / "settings.json", asdict(self.update({"mode": "mute"})))
