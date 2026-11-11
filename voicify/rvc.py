import math

import numpy as np
import onnxruntime as ort
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import resample_poly

from .audio import FRAME
from .config import PROFILES
from .models import get_graph

RATE = 48000
CONTEXT = RATE * 24 // 100
LOOKAHEAD = FRAME * 2
EDGE = RATE * 3 // 1000
BLEND = RATE * 12 // 1000
SEARCH = RATE * 5 // 1000
F0_WINDOW = 640
F0_HOP = 160
RVC_INPUTS = (
    ("phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"),
    ("feats", "p_len", "pitch", "pitchf", "sid"),
    ("feats", "p_len", "sid"),
    ("phone", "phone_lengths", "ds", "rnd"),
)


def resample(samples, source, target):
    divisor = math.gcd(source, target)
    return resample_poly(samples, target // divisor, source // divisor)


def estimate_pitch(samples, frames):
    padded = np.pad(samples, F0_WINDOW // 2)
    windows = np.lib.stride_tricks.sliding_window_view(padded, F0_WINDOW)[::F0_HOP][:frames]
    windows = windows - windows.mean(axis=1, keepdims=True)
    spectrum = rfft(windows, n=next_fast_len(F0_WINDOW * 2))
    lag = np.arange(1, 321)
    correlation = irfft(np.abs(spectrum) ** 2)[:, lag]
    energy = np.cumsum(np.pad(windows ** 2, ((0, 0), (1, 0))), axis=1)
    difference = np.maximum(0, energy[:, -lag - 1] + energy[:, -1:] - energy[:, lag] - 2 * correlation)
    normalized = difference * lag / np.maximum(np.cumsum(difference, axis=1), 1e-10)
    minima = (normalized[:, 1:-1] <= normalized[:, :-2]) & (normalized[:, 1:-1] < normalized[:, 2:])
    candidates = minima & (normalized[:, 1:-1] < .19) & (lag[1:-1] >= 27)
    voiced = candidates.any(axis=1) & (energy[:, -1] > 1e-5)
    positions = candidates.argmax(axis=1) + 1
    row = np.arange(len(windows))
    left, center, right = (normalized[row, positions + offset] for offset in (-1, 0, 1))
    fraction = np.clip((left - right) / np.maximum(2 * (left + right - 2 * center), 1e-8), -.5, .5)
    return np.where(voiced, 16000 / (lag[positions] + fraction), 0).astype(np.float32)


def session(data, backend):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8 if backend == "CPU" else 2
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if backend == "CUDA":
        ort.preload_dlls(directory="")
        providers.insert(0, ("CUDAExecutionProvider", {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
        }))
    result = ort.InferenceSession(data, sess_options=options, providers=providers)
    result.disable_fallback()
    if backend == "CUDA" and "CUDAExecutionProvider" not in result.get_providers():
        raise RuntimeError("CUDA could not initialize. Check the NVIDIA driver and CUDA runtime.")
    return result


class RVC:
    def __init__(self, voice, backend, assets=None):
        if assets is None:
            encoder_key = voice.get("cuda_encoder", voice["encoder"]) if backend == "CUDA" else voice["encoder"]
            assets = get_graph(encoder_key), get_graph(voice["voice"])
        encoder, synth = assets
        self.encoder = session(encoder, backend)
        self.synth = session(synth, backend)
        self.rate = voice["rate"]
        self.backend = backend
        self.speaker = np.array([voice["speaker"]], dtype=np.int64)
        self.rng = np.random.default_rng()
        self.noise = None
        self.noise_frame = None
        encoder_inputs = {item.name: item for item in self.encoder.get_inputs()}
        self.attention_mask = encoder_inputs.pop("attention_mask", None)
        if len(encoder_inputs) != 1:
            raise ValueError("Encoder must accept PCM and an optional attention mask")
        self.encoder_input = next(iter(encoder_inputs.values()))
        if self.encoder_input.type != "tensor(float)" or len(self.encoder_input.shape) not in (2, 3):
            raise ValueError("Encoder must take mono 16 kHz float PCM as [1,T] or [1,1,T]")
        if self.attention_mask is not None and self.attention_mask.type not in ("tensor(int64)", "tensor(bool)"):
            raise ValueError("Encoder attention mask must be int64 or bool")
        self.inputs = {item.name: item for item in self.synth.get_inputs()}
        self.signature = next((names for names in RVC_INPUTS if set(names) == self.inputs.keys()), None)
        if self.signature is None or len(self.synth.get_outputs()) != 1:
            raise ValueError("Unsupported RVC ONNX input or output signature")
        features = self.inputs[self.signature[0]]
        if len(features.shape) != 3 or features.type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError("RVC must take a float feature sequence")
        self.width = features.shape[-1] if features.shape[-1] in (256, 768) else None
        self.dtype = np.float16 if features.type == "tensor(float16)" else np.float32

    def warmup(self):
        for size in PROFILES[self.backend].values():
            length = CONTEXT + LOOKAHEAD + size * RATE // 1000
            samples = np.sin(np.arange(length, dtype=np.float32) * (2 * np.pi * 180 / RATE)) * .05
            self.render(samples, 0)

    def latent(self, channels, frames, start):
        previous = self.noise
        overlap = 0
        if start is not None and self.noise_frame is not None and previous.shape[1] == channels:
            advance = start - self.noise_frame
            if 0 < advance < previous.shape[2]:
                overlap = min(frames, previous.shape[2] - advance)
        tail = self.rng.standard_normal((1, channels, frames - overlap), dtype=np.float32)
        noise = np.concatenate((previous[:, :, advance:advance + overlap], tail), axis=2) if overlap else tail
        self.noise, self.noise_frame = noise, start
        return noise.astype(self.dtype)

    def render(self, samples, semitones, *, start=None):
        pcm = resample(samples, RATE, 16000)
        frames = len(pcm) // F0_HOP
        shape = (1,) * (len(self.encoder_input.shape) - 1) + (-1,)
        feeds = {self.encoder_input.name: pcm.reshape(shape)}
        if self.attention_mask is not None:
            dtype = np.int64 if self.attention_mask.type == "tensor(int64)" else bool
            feeds["attention_mask"] = np.ones((1, len(pcm)), dtype=dtype)
        outputs = self.encoder.run(None, feeds)
        features = next((item for item in outputs if item.ndim == 3 and item.shape[0] == 1
                         and item.shape[-1] in (256, 768) and self.width in (None, item.shape[-1])), None)
        if features is None or abs(features.shape[1] * 2 - frames) > 4:
            raise ValueError("Encoder must return matching 50 Hz RVC content features")
        features = np.repeat(features, 2, axis=1)
        features = np.pad(features, ((0, 0), (0, max(0, frames - features.shape[1])), (0, 0)), mode="edge")
        phone, length = self.signature[:2]
        feeds = {phone: features[:, :frames].astype(self.dtype), length: np.array([frames], dtype=np.int64)}
        feeds["sid" if "sid" in self.inputs else "ds"] = self.speaker
        if "pitch" in self.inputs:
            pitch = estimate_pitch(pcm, frames) * 2 ** (semitones / 12)
            mel = 1127 * np.log1p(pitch / 700)
            low, high = 1127 * np.log1p(np.array([50, 1100]) / 700)
            feeds["pitch"] = np.rint(np.clip(1 + (mel - low) * 254 / (high - low), 1, 255)).astype(np.int64)[None]
            feeds["pitchf"] = pitch[None].astype(np.float32)
        if "rnd" in self.inputs:
            channels = self.inputs["rnd"].shape[1]
            channels = channels if isinstance(channels, int) and 1 <= channels <= 512 else 192
            feeds["rnd"] = self.latent(channels, frames, start)
        output = self.synth.run(None, feeds)[0].squeeze()
        if output.ndim != 1 or abs(len(output) - frames * self.rate // 100) > self.rate // 100 or not np.isfinite(output).all():
            raise ValueError("RVC output does not match the declared voice rate")
        return resample(output.astype(np.float32), self.rate, RATE)


class Converter:
    def __init__(self, engine):
        self.engine = engine
        self.context = np.zeros(CONTEXT, dtype=np.float32)
        self.last_sample = 0.0
        self.last_slope = 0.0
        phase = np.linspace(0, 1, EDGE, dtype=np.float32)
        self.edge = (1 - phase) ** 2 * (1 + 2 * phase)
        self.slope_edge = np.arange(EDGE, dtype=np.float32) * (1 - phase) ** 2
        self.was_active = False
        self.frame = 0
        self.tail = np.empty(0, dtype=np.float32)
        self.fade = np.sin(np.linspace(0, np.pi / 2, BLEND, dtype=np.float32)) ** 2

    def convert(self, block, lookahead, pitch, *, envelope):
        window = np.concatenate((self.context, block, lookahead))
        self.context = window[len(block):len(block) + CONTEXT].copy()
        start = self.frame
        self.frame += len(block) * 100 // RATE
        if not np.any(envelope) or not np.any(block):
            self.last_sample = 0.0
            self.was_active = False
            self.tail = self.tail[:0]
            return np.zeros_like(block)

        rendered = self.engine.render(window, pitch, start=start)
        offset = CONTEXT
        aligned = False
        if self.was_active and envelope[0] > 0 and len(self.tail) == BLEND:
            candidates = rendered[CONTEXT - SEARCH:CONTEXT + SEARCH + BLEND]
            scores = self.alignment(candidates)
            best = int(np.argmax(scores))
            if scores[best] >= .75:
                offset += best - SEARCH
                aligned = True
        audio = rendered[offset:offset + len(block)].copy()
        if len(audio) != len(block):
            raise ValueError("Voice returned a short audio block")
        if aligned:
            audio[:BLEND] = self.tail * (1 - self.fade) + audio[:BLEND] * self.fade
        if self.was_active and envelope[0] > 0:
            jump = self.last_sample + self.last_slope - audio[0]
            slope = np.clip(self.last_slope - (audio[1] - audio[0]), -.001, .001)
            audio[:EDGE] += jump * self.edge + slope * self.slope_edge

        end = offset + len(block)
        self.tail = rendered[end:end + BLEND].copy()
        self.was_active = bool(envelope[-1] > 0)
        self.last_sample = float(audio[-1])
        self.last_slope = float(audio[-1] - audio[-2])
        return audio * envelope

    def alignment(self, candidates):
        # only blend predictions that agree on the shape of the sound
        tail = self.tail - self.tail.mean()
        energy = np.dot(tail, tail)
        if energy < 1e-6 or len(candidates) != BLEND + 2 * SEARCH:
            return np.array([-1.0])
        sums = np.convolve(candidates, np.ones(BLEND), mode="valid")
        power = np.convolve(candidates ** 2, np.ones(BLEND), mode="valid")
        scale = np.sqrt(energy * np.maximum(power - sums ** 2 / BLEND, 1e-8))
        return np.correlate(candidates, tail) / scale - .04 * abs(np.arange(len(scale)) - SEARCH) / SEARCH
