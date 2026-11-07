import collections
import fcntl
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from . import audio
from .config import PROFILES, Settings
from .models import read_voice
from .rvc import LOOKAHEAD, Converter, RVC, ort


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.settings = Settings.load()
        self.status = {"state": "Starting", "registered": False, "devices": [], "input_level": 0,
                       "output_level": 0, "processing_ms": None, "buffer_ms": None, "backend": None, "drops": 0,
                       "message": "", "gpu": None}
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def update(self, changes):
        with self.lock:
            updated = self.settings.update(changes)
            if updated.model and updated.model != self.settings.model:
                read_voice(updated.model)
            if updated != self.settings:
                updated.save()
                self.settings = updated

    def snapshot(self):
        with self.lock:
            return dict(self.status, settings=asdict(self.settings))

    def set_status(self, **changes):
        with self.lock:
            self.status.update(changes)

    def run(self):
        output = None
        with ThreadPoolExecutor(max_workers=1) as worker:
            pipeline = Pipeline(worker)
            next_scan = next_gpu = 0
            next_frame = time.monotonic()
            sources = []
            graph_available = False
            registered = False
            sink = ""
            try:
                while not self.closing.is_set():
                    tick = time.monotonic()
                    if tick >= next_scan:
                        next_scan = tick + 1.0
                        try:
                            sources, registered = audio.devices()
                            graph_available = True
                            sink = audio.playback_sink()
                            self.set_status(devices=sources, registered=registered, message="")
                        except (OSError, ValueError, subprocess.SubprocessError):
                            sources, registered = [], False
                            graph_available = False
                            self.set_status(devices=[], registered=False, message="PipeWire is unavailable")
                    if output is None or output.poll() is not None:
                        audio.stop(output)
                        output = None
                        if graph_available:
                            try:
                                output = audio.virtual_mic()
                                fd = output.stdin.fileno()
                                os.set_blocking(fd, False)
                                fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 8192)
                            except OSError as exc:
                                self.set_status(message=str(exc))
                    if tick >= next_gpu:
                        next_gpu = tick + 5
                        worker_gpu(self.set_status)
                    with self.lock:
                        settings = self.settings
                    source = next((item for item in sources if item["name"] == settings.device), None)
                    if not settings.device and sources:
                        source = sources[0]
                    block, status = pipeline.step(settings, source if registered and output else None, sink)
                    if output is not None and output.poll() is None:
                        try:
                            os.write(output.stdin.fileno(), block)
                        except BlockingIOError:
                            pass
                        except BrokenPipeError:
                            audio.stop(output)
                            output = None
                    if not registered:
                        status["state"] = "Waiting for PipeWire"
                    self.set_status(**status)
                    next_frame += .02
                    now = time.monotonic()
                    if next_frame < now - .02:
                        next_frame = now
                    self.closing.wait(max(0, next_frame - now))
            finally:
                pipeline.reset()
                audio.stop(output)


def worker_gpu(report):
    def query():
        try:
            command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                       "--format=csv,noheader,nounits"]
            line = subprocess.check_output(command, timeout=1, stderr=subprocess.DEVNULL, text=True).splitlines()[0]
            utilization, used, total = (int(value.strip()) for value in line.split(","))
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            report(gpu=None)
        else:
            report(gpu={"utilization": utilization, "used": used, "total": total})
    threading.Thread(target=query, daemon=True).start()


def load_converter(model, backend):
    if backend == "CUDA" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA runtime is missing. Run ./setup cuda.")
    target = backend
    if target == "Auto":
        target = "CUDA" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPU"
    voice = read_voice(model)
    try:
        engine = RVC(voice, target)
        engine.warmup()
    except Exception:
        if backend != "Auto" or target != "CUDA":
            raise
        target = "CPU"
        engine = RVC(voice, target)
        engine.warmup()
    return engine, target


def convert(converter, block, ahead, pitch):
    started = time.monotonic()
    samples = converter.convert(block, ahead, pitch)
    return samples, (time.monotonic() - started) * 1000


class Pipeline:
    def __init__(self, worker):
        self.worker = worker
        self.capture = None
        self.key = None
        self.job = None
        self.load = None
        self.converter = None
        self.warm = None
        self.warm_key = None
        self.backend = None
        self.input = collections.deque()
        self.output = collections.deque()
        self.playing = False
        self.ready_at = 0.0
        self.gate = audio.Gate()
        self.error = ""
        self.ms = None
        self.drops = 0
        self.last_frame = 0.0

    def reset(self):
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        for task in (self.job, self.load):
            if task is not None:
                task.cancel()
        self.job = self.load = self.converter = self.backend = None
        self.input.clear()
        self.output.clear()
        self.playing = False
        self.ready_at = 0.0
        self.gate = audio.Gate()
        self.error = ""
        self.ms = None

    def step(self, settings, source, sink=""):
        serial = source["serial"] if source else None
        reference = sink if settings.cleanup else ""
        key = (serial, settings.mode, settings.model, settings.backend, settings.profile, settings.pitch,
               settings.cleanup, reference, settings.noise_suppression)
        if key != self.key:
            self.reset()
            engine_key = (settings.model, settings.backend)
            if settings.mode != "convert" or engine_key != self.warm_key:
                self.warm = None
                self.warm_key = engine_key
            self.key = key
        status = {"state": "Muted", "input_level": 0.0, "output_level": 0.0, "backend": self.backend,
                  "processing_ms": self.ms, "buffer_ms": None, "drops": self.drops, "fault": self.error,
                  "cleanup_active": False}
        if settings.mode == "mute":
            return audio.SILENCE, status
        if not serial:
            return audio.SILENCE, dict(status, state="Waiting for microphone")
        if self.error:
            return audio.SILENCE, dict(status, state="Conversion paused")
        if settings.mode == "convert":
            if self.converter is None:
                if self.warm is not None:
                    engine, self.backend = self.warm
                    self.converter = Converter(engine)
                elif self.load is None:
                    self.load = self.worker.submit(load_converter, settings.model, settings.backend)
            if self.load is not None:
                if not self.load.done():
                    return audio.SILENCE, dict(status, state="Loading voice")
                try:
                    self.warm = self.load.result()
                    engine, self.backend = self.warm
                    self.converter = Converter(engine)
                except Exception as exc:
                    self.error = str(exc)
                    return audio.SILENCE, dict(status, state="Conversion paused", fault=self.error)
                self.load = None
        try:
            if self.capture is None:
                self.capture = audio.Capture(serial, reference, noise_suppression=settings.noise_suppression)
                self.last_frame = time.monotonic()
            samples = self.capture.frame(.012)
            if samples is None and time.monotonic() - self.last_frame > 1:
                raise TimeoutError("No microphone frames received")
        except (OSError, EOFError) as exc:
            self.reset()
            return audio.SILENCE, dict(status, state="Reconnecting microphone", fault=str(exc))
        rms = 0.0
        if samples is not None:
            self.last_frame = time.monotonic()
            np.nan_to_num(samples, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(samples * (settings.input_gain / 100), -1, 1, out=samples)
            rms = audio.level(samples)
            self.gate.apply(samples, settings.gate, rms)
        if self.converter is not None:
            samples = self.infer(samples, settings)
        state = "Converting" if settings.mode == "convert" else "Direct"
        if self.error:
            state = "Conversion paused"
        elif self.converter is not None and not self.playing:
            state = "Buffering voice"
        status.update(state=state, input_level=rms, processing_ms=self.ms, drops=self.drops,
                      backend=self.backend, fault=self.error or self.capture.cleanup_fault or (
                          "Echo and noise cleanup is waiting for a default output device."
                          if settings.cleanup and not reference else ""
                      ),
                      cleanup_active=self.capture.filter is not None,
                      buffer_ms=PROFILES[self.backend][settings.profile] if self.backend else None)
        if samples is None:
            return audio.SILENCE, status
        np.clip(samples * (settings.output_gain / 100), -1, 1, out=samples)
        status["output_level"] = audio.level(samples)
        return samples.astype("<f4", copy=False).tobytes(), status

    def infer(self, samples, settings):
        block_frames = PROFILES[self.backend][settings.profile] // 20
        lookahead_frames = LOOKAHEAD // audio.FRAME
        if self.job is not None and self.job.done():
            try:
                output, self.ms = self.job.result()
            except Exception as exc:
                self.output.clear()
                self.input.clear()
                self.playing = False
                self.ready_at = 0.0
                if settings.backend == "Auto" and self.backend == "CUDA":
                    self.converter = self.warm = None
                    self.load = self.worker.submit(load_converter, settings.model, "CPU")
                    self.job = None
                    return None
                self.error = str(exc)
                return None
            self.job = None
            if len(self.output) + block_frames > 2 * block_frames:
                self.output.clear()
                self.drops += 1
            self.output.extend(np.split(output, block_frames))
            if not self.playing and not self.ready_at:
                self.ready_at = time.monotonic() + .04
        if samples is not None:
            self.input.append(samples)
        limit = block_frames + lookahead_frames
        if len(self.input) > limit:
            # keep the freshest speech if inference falls behind
            self.input.popleft()
            self.drops += 1
        if len(self.input) == limit and self.job is None:
            block = np.concatenate([self.input.popleft() for _ in range(block_frames)])
            ahead = np.concatenate(self.input)
            self.job = self.worker.submit(convert, self.converter, block, ahead, settings.pitch)
        if self.output and (self.playing or time.monotonic() >= self.ready_at):
            self.playing = True
            return self.output.popleft()
        if self.playing:
            self.playing = False
            self.ready_at = 0.0
            self.drops += 1
        return None
