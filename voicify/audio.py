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

    def apply(self, samples, threshold, rms):
        if threshold == -80:
            self.gain = 0.0
            self.hold = 0
            return samples
        # quiet syllables need less level to keep an open gate open
        trigger = threshold - 8 if self.hold else threshold
        if rms >= 10 ** (trigger / 20):
            self.hold = 7
        target = float(self.hold > 0)
        self.hold = max(0, self.hold - 1)
        ramp = self.attack if target > self.gain else self.ramp
        samples *= self.gain + (target - self.gain) * ramp
        self.gain = target
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


def preview():
    from .control import request

    status = request("status")
    if not status["registered"] or status["settings"]["mode"] == "mute":
        raise ValueError("Start Direct or Convert before recording a preview")
    sink = subprocess.check_output(["pactl", "get-default-sink"], timeout=1, text=True).strip()
    target = headphone_sink(sink)
    samples = RATE * 4
    recording = stream(True, MIC, PINNED, samples=samples)
    try:
        clip, _ = recording.communicate(timeout=7)
    finally:
        stop(recording)
    if len(clip) != samples * 4:
        raise ValueError("The virtual microphone stopped before the preview finished")
    if not np.any(np.frombuffer(clip, dtype="<f4")):
        raise ValueError("The preview was silent. Check the input meter and try again.")
    if headphone_sink(sink) != target:
        raise ValueError("Headphones changed during recording. Try the preview again.")

    # play only after capture has closed, so the preview can't feed itself
    print("Playing preview…", flush=True)
    playback = stream(False, target, PINNED)
    try:
        playback.communicate(input=clip, timeout=7)
        if playback.returncode:
            raise ValueError("Could not play the preview through your headphones")
    finally:
        stop(playback)
