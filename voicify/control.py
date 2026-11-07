import json
import socket

from .config import socket_path

MAX_REQUEST = 4096
MAX_REPLY = 262144


def request(action, **payload):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(.35)
        connection.connect(str(socket_path()))
        connection.sendall(json.dumps({"action": action, **payload}, allow_nan=False).encode() + b"\n")
        with connection.makefile("rb") as file:
            data = file.readline(MAX_REPLY)
    if not data.endswith(b"\n"):
        raise OSError("Incomplete reply from the audio service")
    reply = json.loads(data)
    if "error" in reply:
        raise ValueError(reply["error"])
    return reply
