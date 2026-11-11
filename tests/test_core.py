import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import onnx

from voicify.audio import FRAME, RATE, Gate, headphone_sink, level
from voicify.config import Settings
from voicify.models import asset, validate_graph
from voicify.pitch import PitchShift
from voicify.rvc import CONTEXT, LOOKAHEAD, Converter, estimate_pitch


class CoreTests(unittest.TestCase):
    def test_pitch_only_shifts_both_directions_and_bypasses_at_zero(self):
        samples = np.sin(2 * np.pi * 250 * np.arange(RATE) / RATE).astype(np.float32)
        for semitones, expected in ((-12, 125), (0, 250), (12, 500)):
            shift = PitchShift()
            result = np.concatenate([
                shift.process(frame, semitones) for frame in samples.reshape(-1, FRAME)
            ])
            if semitones == 0:
                np.testing.assert_array_equal(result, samples)
            steady = result[RATE // 5:]
            spectrum = np.fft.rfft(steady * np.hanning(len(steady)))
            peak = np.argmax(abs(spectrum)) * RATE / len(steady)
            self.assertAlmostEqual(peak, expected, delta=2)

    def test_configuration_rejects_unselected_voice_and_invalid_gain(self):
        for settings in ({"mode": "convert"}, {"input_gain": True}, {"pitch": 35},
                         {"model": "../voice"}, {"backend": "GPU"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Settings().update(settings)

    def test_gate_closes_and_retains_loud_signal(self):
        quiet = np.full(FRAME, .00001, dtype=np.float32)
        loud = np.full(FRAME, .3, dtype=np.float32)
        gate = Gate()
        self.assertFalse(np.any(gate.apply(quiet, -55, level(quiet))))
        self.assertGreater(level(gate.apply(loud, -55, level(loud))), .1)

    def test_gate_keeps_soft_syllables_but_does_not_reopen_on_them(self):
        gate = Gate()
        soft = np.full(FRAME, .005, dtype=np.float32)
        speech = np.full(FRAME, .03, dtype=np.float32)
        room = np.full(FRAME, .001, dtype=np.float32)
        self.assertFalse(np.any(gate.apply(soft.copy(), -40, level(soft))))
        gate.apply(speech, -40, level(speech))
        for _ in range(12):
            np.testing.assert_array_equal(gate.apply(soft.copy(), -40, level(soft)), soft)
        for _ in range(8):
            tail = gate.apply(room.copy(), -40, level(room))
        self.assertFalse(np.any(tail))
        self.assertFalse(np.any(gate.apply(soft.copy(), -40, level(soft))))

    def test_pitch_tracks_tone_and_silence(self):
        samples = np.sin(np.arange(4800) * 2 * np.pi * 215 / 16000)
        estimated = estimate_pitch(samples, 30)
        self.assertLess(abs(np.median(estimated[4:-4]) - 215), 1)
        self.assertFalse(np.any(estimate_pitch(samples * 0, 30)))

    def test_overlap_returns_one_block_and_skips_silence(self):
        class Engine:
            def render(self, samples, pitch, *, start=None):
                return samples.copy()

        converter = Converter(Engine())
        block = np.ones(FRAME * 4, np.float32)
        out = converter.convert(block, np.ones(LOOKAHEAD, np.float32), 0, envelope=block)
        self.assertEqual(out.shape, (FRAME * 4,))
        self.assertEqual(out[-1], 1)
        self.assertEqual(converter.context.shape, (CONTEXT,))

    def test_chunk_join_does_not_mix_in_the_previous_prediction(self):
        class Voice:
            def render(self, samples, pitch, *, start=None):
                out = samples.copy()
                out[-LOOKAHEAD:] = .9
                return out

        converter = Converter(Voice())
        block = np.full(FRAME * 4, .2, dtype=np.float32)
        ahead = np.full(LOOKAHEAD, .2, dtype=np.float32)
        converter.convert(block, ahead, 0, envelope=np.ones_like(block))
        result = converter.convert(block, ahead, 0, envelope=np.ones_like(block))
        np.testing.assert_allclose(result, block)

    def test_converter_hears_quiet_context_without_sending_it_to_the_mic(self):
        class Voice:
            def render(self, samples, pitch, *, start=None):
                self.input = samples
                return np.ones_like(samples)

        voice = Voice()
        converter = Converter(voice)
        block = np.full(FRAME * 4, .005, dtype=np.float32)
        ahead = np.full(LOOKAHEAD, .003, dtype=np.float32)
        gain = np.repeat(np.array([1, 0, .4, 0], dtype=np.float32), FRAME)
        result = converter.convert(block, ahead, 0, envelope=gain)

        np.testing.assert_array_equal(voice.input[CONTEXT:], np.concatenate((block, ahead)))
        np.testing.assert_array_equal(result, gain)
        self.assertFalse(np.any(converter.convert(block, ahead, 0, envelope=np.zeros_like(block))))

    def test_monitor_accepts_headphones_and_rejects_speakers(self):
        sink = {"name": "built-in", "active_port": "analog-output-headphones",
                "properties": {"object.serial": "12", "device.form_factor": "internal"}}
        with patch("voicify.audio.subprocess.check_output", side_effect=lambda *a, **kw: json.dumps([sink])):
            self.assertEqual(headphone_sink("built-in"), "12")
            sink["active_port"] = "analog-output-speaker"
            with self.assertRaisesRegex(ValueError, "Select headphones"):
                headphone_sink("built-in")

    def test_model_rejects_external_tensors_before_loading(self):
        external = onnx.TensorProto(name="weights", data_type=onnx.TensorProto.FLOAT, dims=[1])
        external.data_location = onnx.TensorProto.EXTERNAL
        external.external_data.add(key="location", value="../../private.bin")
        graph = onnx.helper.make_graph([], "external", [], [], [external])
        model = onnx.helper.make_model(graph)
        with self.assertRaisesRegex(ValueError, "External ONNX"):
            validate_graph(model.SerializeToString())

    def test_model_rejects_pickle_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            path.write_bytes(b"not an ONNX graph")
            with self.assertRaisesRegex(ValueError, "standalone"):
                asset(path)
            renamed = path.rename(path.with_suffix(".onnx"))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                asset(renamed, "0" * 64)


if __name__ == "__main__":
    unittest.main()
