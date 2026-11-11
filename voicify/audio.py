import json
import os
import selectors
import subprocess
import sys
import time

import numpy as np

RATE = 48000
FRAME = RATE // 50
FRAME_BYTES = FRAME * 4
SILENCE = bytes(FRAME_BYTES)
MIC = "voicify.virtual.microphone"
LABEL = "Voicify Virtual Microphone"
PINNED = {
    "node.dont-fallback": True,
    "node.dont-reconnect": True,
    "node.dont-move": True,
}


def devices():
    graph = json.loads(subprocess.check_output(["pw-dump"], timeout=2, stderr=subprocess.DEVNULL))
    result = []
    for node in graph:
        props = node.get("info", {}).get("props", {})
        name = props.get("node.name", "")
        if (node["type"] == "PipeWire:Interface:Node" and props.get("media.class") == "Audio/Source"
                and props.get("device.id") is not None and props.get("device.api") in ("alsa", "bluez5")
                and name and not name.endswith(".monitor")):
            result.append({"name": name, "label": props.get("node.description", name),
                           "serial": str(props["object.serial"])})
    default = ""
    try:
        default = subprocess.check_output(["pactl", "get-default-source"], timeout=1,
                                          stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    registered = any(node.get("info", {}).get("props", {}).get("node.name") == MIC for node in graph)
    return sorted(result, key=lambda item: (item["name"] != default, item["label"])), registered


def pipewire_process(command, **options):
    return subprocess.Popen([sys.executable, "-m", "voicify.child", str(os.getpid()), *command], **options)


def stream(record, target, properties, samples=None):
    direction = "-r" if record else "-p"
    limit = ["--sample-count", str(samples)] if samples is not None else []
    return pipewire_process(
        ["pw-cat", direction, "--raw", "--rate", str(RATE), "--channels", "1", "--channel-map", "MONO",
         "--format", "f32", "--latency", "20ms", "--target", target, "-P", json.dumps(properties), *limit, "-"],
        stdin=subprocess.DEVNULL if record else subprocess.PIPE,
        stdout=subprocess.PIPE if record else subprocess.DEVNULL,
        bufsize=0,
    )


def virtual_mic():
    return stream(False, "0", {
        "media.class": "Audio/Source",
        "node.name": MIC,
        "node.description": LABEL,
        "node.virtual": True,
        "node.always-process": True,
        "priority.session": 0,
    })


def stop(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=.4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for pipe in (process.stdin, process.stdout):
        if pipe is not None:
            pipe.close()


def playback_sink():
    try:
        return subprocess.check_output(["pactl", "get-default-sink"], timeout=1,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def echo_filter(serial, sink, noise_suppression=False):
    name = f"voicify.clean.{os.getpid()}"
    options = {
        "monitor.mode": True,
        "library.name": "aec/libspa-aec-webrtc",
        "audio.rate": RATE,
        "audio.channels": 1,
        "audio.position": ["MONO"],
        "aec.args": {"webrtc.noise_suppression": noise_suppression, "webrtc.high_pass_filter": True,
                     "webrtc.gain_control": False},
        "capture.props": {**PINNED, "target.object": serial},
        "sink.props": {**PINNED, "target.object": sink},
        "source.props": {"node.name": name, "node.description": "Voicify Clean Mic",
                         "node.virtual": True, "priority.session": 0},
    }
    process = pipewire_process(
        ["pw-cli", "-m", "load-module", "libpipewire-module-echo-cancel", json.dumps(options)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and process.poll() is None:
            graph = json.loads(subprocess.check_output(["pw-dump"], timeout=.5, stderr=subprocess.DEVNULL))
            for node in graph:
                props = node.get("info", {}).get("props", {})
                if (props.get("node.name") == name and props.get("media.class") == "Audio/Source"
                        and "object.serial" in props):
                    return process, str(props["object.serial"])
            time.sleep(.04)
        raise OSError("PipeWire echo filter did not start")
    except (OSError, ValueError, subprocess.SubprocessError):
        stop(process)
        raise


class Capture:
    def __init__(self, serial, sink="", *, noise_suppression=False):
        self.filter = None
        self.cleanup_fault = ""
        target = serial
        if sink:
            try:
                self.filter, target = echo_filter(serial, sink, noise_suppression)
            except (OSError, ValueError, subprocess.SubprocessError):
                self.cleanup_fault = "Echo and noise cleanup unavailable. Using the original mic signal."
        try:
            self.process = stream(True, target, PINNED)
            self.target = target
        except OSError:
            stop(self.filter)
            raise
        self.fd = self.process.stdout.fileno()
        self.reader = selectors.DefaultSelector()
        self.reader.register(self.fd, selectors.EVENT_READ)
        self.buffer = bytearray()

    def frame(self, timeout):
        end = time.monotonic() + timeout
        while len(self.buffer) < FRAME_BYTES:
            if not self.reader.select(max(0, end - time.monotonic())):
                return None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                raise EOFError("Microphone disconnected")
            self.buffer.extend(chunk)
        if len(self.buffer) >= FRAME_BYTES * 4:
            # drop old frames before they turn into lag
            discard = (len(self.buffer) // FRAME_BYTES - 1) * FRAME_BYTES
            del self.buffer[:discard]
        block = bytes(self.buffer[:FRAME_BYTES])
        del self.buffer[:FRAME_BYTES]
        return np.frombuffer(block, dtype="<f4").copy()

    def close(self):
        self.reader.close()
        stop(self.process)
        stop(self.filter)


def level(samples):
    return float(np.sqrt(np.mean(np.square(samples))))


class Gate:
    def __init__(self):
        self.gain = 0.0
        self.hold = 0
        self.ramp = np.linspace(0, 1, FRAME, dtype=np.float32)
        self.attack = np.minimum(self.ramp * 4, 1)

    def envelope(self, threshold, rms):
        if threshold == -80:
            self.gain = 0.0
            self.hold = 0
            return 1.0
        # quiet syllables need less level to keep an open gate open
        trigger = threshold - 8 if self.hold else threshold
        if rms >= 10 ** (trigger / 20):
            self.hold = 7
        target = float(self.hold > 0)
        self.hold = max(0, self.hold - 1)
        previous = self.gain
        self.gain = target
        if previous == target:
            return target
        ramp = self.attack if target > previous else self.ramp
        return previous + (target - previous) * ramp

    def apply(self, samples, threshold, rms):
        samples *= self.envelope(threshold, rms)
        return samples


def headphone_sink(name):
    sinks = json.loads(subprocess.check_output(
        ["pactl", "-f", "json", "list", "sinks"], timeout=1, stderr=subprocess.DEVNULL,
    ))
    sink = next((item for item in sinks if item["name"] == name), {})
    props = sink.get("properties", {})
    port = sink.get("active_port") or ""
    if isinstance(port, dict):
        port = port.get("name", "")
    port = port.lower()
    form = props.get("device.form_factor", "")
    headphone_port = any(kind in port for kind in ("headphone", "headset"))
    speaker_port = any(kind in port for kind in ("speaker", "lineout", "line-out"))
    headphone_device = form in ("headphone", "headphones", "headset")
    if speaker_port or not (headphone_port or headphone_device):
        raise ValueError("Select headphones as your system output before monitoring")
    return str(props["object.serial"])


def preview(compare=False):
    from .control import request

    status = request("status")
    if not status["registered"] or status["settings"]["mode"] == "mute":
        raise ValueError("Start Direct or Convert before recording a preview")
    sink = subprocess.check_output(["pactl", "get-default-sink"], timeout=1, text=True).strip()
    target = headphone_sink(sink)
    samples = RATE * 4
    source = status.get("capture_target") if compare else MIC
    if not source or (compare and not status["settings"]["model"]):
        raise ValueError("Choose a voice and start Direct or Convert before comparing")
    recording = stream(True, source, PINNED, samples=samples)
    try:
        clip, _ = recording.communicate(timeout=7)
    finally:
        stop(recording)
    if len(clip) != samples * 4:
        raise ValueError("The virtual microphone stopped before the preview finished")
    if not np.any(np.frombuffer(clip, dtype="<f4")):
        raise ValueError("The preview was silent. Check the input meter and try again.")
    if compare:
        print("Converting the recorded take…", flush=True)
        direct, converted = compare_take(np.frombuffer(clip, dtype="<f4"), status["settings"])
        for name, take in (("Direct", direct), ("Convert", converted)):
            play_preview(sink, target, take.astype("<f4", copy=False).tobytes(), f"Playing {name}…")
    else:
        play_preview(sink, target, clip, "Playing preview…")


def compare_take(samples, settings):
    from .config import PROFILES
    from .engine import load_converter
    from .rvc import LOOKAHEAD, Converter

    model, backend = load_converter(settings["model"], settings["backend"])
    converter = Converter(model)
    size = RATE * PROFILES[backend][settings["profile"]] // 1000
    frames = np.pad(samples, (0, (-len(samples)) % size + LOOKAHEAD))
    frames = np.nan_to_num(frames, nan=0, posinf=0, neginf=0)
    np.clip(frames * (settings["input_gain"] / 100), -1, 1, out=frames)
    gate = Gate()
    envelope = np.concatenate([
        np.broadcast_to(gate.envelope(settings["gate"], level(frame)), frame.shape)
        for frame in frames.reshape(-1, FRAME)
    ], dtype=np.float32)
    converted = [
        converter.convert(frames[i:i + size], frames[i + size:i + size + LOOKAHEAD],
                          settings["pitch"], envelope=envelope[i:i + size])
        for i in range(0, len(frames) - LOOKAHEAD, size)
    ]
    output_gain = settings["output_gain"] / 100
    return (np.clip(frames[:len(samples)] * envelope[:len(samples)] * output_gain, -1, 1),
            np.clip(np.concatenate(converted)[:len(samples)] * output_gain, -1, 1))


def play_preview(sink, target, clip, message):
    if headphone_sink(sink) != target:
        raise ValueError("Headphones changed during recording. Try the preview again.")

    # play after recording so the preview can't feed itself
    print(message, flush=True)
    playback = stream(False, target, PINNED)
    try:
        playback.communicate(input=clip, timeout=7)
        if playback.returncode:
            raise ValueError("Could not play the preview through your headphones")
    finally:
        stop(playback)


def virtual_mic_listeners():
    graph = json.loads(subprocess.check_output(["pw-dump"], timeout=2, stderr=subprocess.DEVNULL))
    nodes = {entry["id"]: entry["info"]["props"] for entry in graph
             if entry.get("type") == "PipeWire:Interface:Node"}
    source = next((identifier for identifier, props in nodes.items() if props.get("node.name") == MIC), None)
    listeners = set()
    for entry in graph:
        if entry.get("type") == "PipeWire:Interface:Link" and entry["info"]["output-node-id"] == source:
            props = nodes.get(entry["info"]["input-node-id"], {})
            name = props.get("application.name") or props.get("node.description") or props.get("node.name")
            if name:
                listeners.add(name)
    return sorted(listeners)
