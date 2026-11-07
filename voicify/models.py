import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

import onnx

from .config import DATA, write_json

MODELS = DATA / "voices"
ASSETS = DATA / "graphs"
MAX_GRAPH = 512 * 1024 * 1024
DIGEST = re.compile(r"[a-f0-9]{64}")
RATES = (32000, 40000, 48000)


def read_asset(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError(f"Invalid file size or type: {path.name}")
        data = file.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError(f"Invalid file size: {path.name}")
    return data


def validate_graph(data):
    graph = onnx.load_model_from_string(data)
    pending = [graph]
    while pending:
        message = pending.pop()
        if isinstance(message, onnx.TensorProto) and (message.external_data or message.data_location):
            raise ValueError("External ONNX tensor data is disabled")
        if isinstance(message, onnx.NodeProto) and message.domain not in ("", "ai.onnx", "com.microsoft"):
            raise ValueError(f"Unsupported ONNX operator domain: {message.domain}")
        for field, value in message.ListFields():
            if field.message_type is not None:
                pending.extend(value if field.is_repeated else [value])
    onnx.checker.check_model(graph)


def asset(path, expected=None):
    if path.suffix.lower() != ".onnx":
        raise ValueError("Choose a standalone .onnx graph")
    data = read_asset(path, MAX_GRAPH)
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    validate_graph(data)
    return digest, data


def put_graph(digest, data):
    ASSETS.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ASSETS, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
            temporary.replace(ASSETS / f"{digest}.onnx")
        finally:
            temporary.unlink(missing_ok=True)
    return digest


def get_graph(digest):
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError("Invalid graph digest")
    return asset(ASSETS / f"{digest}.onnx", digest)[1]


def read_voice(identifier):
    if not DIGEST.fullmatch(identifier):
        raise ValueError("Invalid voice ID")
    voice = json.loads(read_asset(MODELS / f"{identifier}.json", 16384))
    validate_voice(voice)
    return voice


def validate_voice(voice):
    if not isinstance(voice, dict):
        raise ValueError("Invalid voice manifest")
    keys = ("encoder", "voice", "cuda_encoder") if "cuda_encoder" in voice else ("encoder", "voice")
    for key in keys:
        if not isinstance(voice.get(key), str) or not DIGEST.fullmatch(voice[key]):
            raise ValueError(f"Invalid {key} checksum")
    if voice.get("rate") not in RATES or type(voice.get("rate")) is not int:
        raise ValueError("Voice rate must be 32000, 40000 or 48000")
    if type(voice.get("speaker")) is not int or not 0 <= voice["speaker"] < 65536:
        raise ValueError("Invalid speaker ID")
    if not isinstance(voice.get("name"), str) or not 0 < len(voice["name"].strip()) <= 80:
        raise ValueError("Enter a voice name between 1 and 80 characters")
    if not isinstance(voice.get("credit", ""), str) or len(voice.get("credit", "")) > 600:
        raise ValueError("Voice credit is too long")


def add_voice(name, encoder, synthesizer, rate, speaker, credit="", hashes=None, cuda_encoder=None):
    hashes = hashes or {}
    encoder_hash, encoder_data = asset(encoder, hashes.get("encoder"))
    voice_hash, voice_data = asset(synthesizer, hashes.get("voice"))
    voice = {"name": name.strip(), "rate": rate, "speaker": speaker, "credit": credit,
             "encoder": encoder_hash, "voice": voice_hash}
    if cuda_encoder is not None:
        cuda_hash, cuda_data = asset(cuda_encoder, hashes.get("cuda_encoder"))
        voice["cuda_encoder"] = cuda_hash
    validate_voice(voice)
    from .rvc import RVC
    engine = RVC(voice, "CPU", assets=(encoder_data, voice_data))
    engine.warmup()
    voice["pitch_supported"] = "pitch" in engine.inputs
    del engine
    put_graph(encoder_hash, encoder_data)
    put_graph(voice_hash, voice_data)
    if cuda_encoder is not None:
        put_graph(cuda_hash, cuda_data)
    identifier = hashlib.sha256(json.dumps(voice, sort_keys=True).encode()).hexdigest()
    write_json(MODELS / f"{identifier}.json", voice)
    return identifier


def voices():
    result = []
    for path in sorted(MODELS.glob("*.json")):
        try:
            voice = read_voice(path.stem)
        except (OSError, ValueError, TypeError):
            continue
        result.append({"id": path.stem, "name": voice["name"], "credit": voice.get("credit", ""),
                       "pitch_supported": voice.get("pitch_supported", True)})
    return sorted(result, key=lambda item: item["name"].casefold())
