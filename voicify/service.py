import fcntl
import json
import os
import signal
import socket
import stat
import subprocess

from .config import socket_path
from .control import MAX_REQUEST
from .engine import Engine
from .models import edit_voice, read_voice, remove_voice, voices


def handle(engine, message):
    action = message["action"]
    if action == "update":
        changes = message["settings"]
        if not isinstance(changes, dict):
            raise ValueError("Invalid settings")
        engine.update(changes)
    elif action in ("rename_voice", "save_preset", "remove_voice"):
        identifier = message["voice"]
        read_voice(identifier)
        if action == "rename_voice":
            if not isinstance(message["name"], str):
                raise ValueError("Enter a voice name")
            edit_voice(identifier, name=message["name"])
        elif action == "save_preset":
            fields = ("pitch", "input_gain", "output_gain", "gate", "profile")
            settings = engine.snapshot()["settings"]
            edit_voice(identifier, preset={field: settings[field] for field in fields})
        else:
            if engine.snapshot()["settings"]["model"] == identifier:
                engine.update({"mode": "mute", "model": ""})
            remove_voice(identifier)
    elif action == "stop":
        engine.closing.set()
    elif action != "status":
        raise ValueError("Unknown action")
    return dict(engine.snapshot(), voices=voices())


def serve():
    os.umask(0o077)
    path = socket_path()
    path.parent.mkdir(mode=0o700, exist_ok=True)
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError("Voicify requires a private runtime directory")
    with (path.parent / "service.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            listener.listen(8)
            listener.settimeout(.2)
            engine = None
            try:
                engine = Engine()
                for signum in (signal.SIGTERM, signal.SIGINT):
                    signal.signal(signum, lambda *_: engine.closing.set())
                while not engine.closing.is_set():
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with connection:
                        connection.settimeout(.25)
                        try:
                            with connection.makefile("rb") as file:
                                data = file.readline(MAX_REQUEST)
                            if not data.endswith(b"\n"):
                                raise ValueError("Invalid request length")
                            reply = handle(engine, json.loads(data))
                        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
                            reply = {"error": str(exc)}
                        try:
                            connection.sendall(json.dumps(reply).encode() + b"\n")
                        except OSError:
                            pass
            finally:
                path.unlink(missing_ok=True)
                if engine is not None:
                    engine.closing.set()
                    engine.thread.join(timeout=4)
