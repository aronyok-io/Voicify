import numpy as np

from .audio import FRAME, RATE

# Two read heads crossfade before either one wraps.
GRAIN = RATE // 25
MIN_DELAY = 2


class PitchShift:
    def __init__(self):
        self.history = np.zeros(GRAIN + MIN_DELAY + 2, dtype=np.float32)
        self.phase = 0.0
        self.samples = np.arange(FRAME)
        self.position = self.samples + len(self.history)

    def process(self, samples, semitones):
        window = np.concatenate((self.history, samples))
        self.history[:] = window[-len(self.history):]
        if semitones == 0:
            self.phase = 0.0
            return samples

        step = (1 - 2 ** (semitones / 12)) / GRAIN
        first = (self.phase + step * self.samples) % 1
        second = (first + .5) % 1
        self.phase = (self.phase + step * FRAME) % 1
        weight = np.sin(np.pi * first) ** 2
        a = self.read(window, first)
        b = self.read(window, second)
        return (a * weight + b * (1 - weight)).astype(np.float32)

    def read(self, window, phase):
        position = self.position - MIN_DELAY - phase * GRAIN
        index = position.astype(np.intp)
        fraction = position - index
        return window[index] + (window[index + 1] - window[index]) * fraction
