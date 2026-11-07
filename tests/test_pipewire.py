import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from voicify import audio, engine
from voicify.config import Settings

SOURCE = "voicify.integration.fixture"


def wait_for(condition, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = condition()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError("Timed out waiting for PipeWire")


def record(seconds):
    samples = int(seconds * audio.RATE)
    process = subprocess.run(
        ["pw-cat", "-r", "--raw", "--rate", "48000", "--channels", "1", "--channel-map", "MONO",
         "--format", "f32", "--latency", "20ms", "--target", audio.MIC,
         "--sample-count", str(samples), "-"],
        stdout=subprocess.PIPE, timeout=seconds + 4,
    )
    if len(process.stdout) != samples * 4:
        process.check_returncode()
        raise AssertionError("PipeWire returned a short recording")
    return np.frombuffer(process.stdout, "<f4")


@unittest.skipUnless(os.getenv("VOICIFY_TEST_PIPEWIRE"), "requires an idle PipeWire desktop")
class PipeWireTests(unittest.TestCase):
    def test_direct_mute_recovery_and_neural_signal(self):
        self.assertFalse(audio.devices()[1], "Stop any active Voicify service before testing")
        default_mic = subprocess.check_output(["pactl", "get-default-source"])
        default_sink = subprocess.check_output(["pactl", "get-default-sink"])
        devices = audio.devices
        tone = np.sin(np.arange(audio.RATE) * (2 * np.pi * 220 / audio.RATE)).astype("<f4") * .2
        with tempfile.TemporaryFile() as pcm:
            pcm.write(np.tile(tone, 50).tobytes())
            pcm.seek(0)
            source = subprocess.Popen(
                ["pw-cat", "-p", "--target", "0", "--raw", "--rate", "48000", "--channels", "1",
                 "--channel-map", "MONO", "--format", "f32", "--latency", "20ms", "-P",
                 json.dumps({"media.class": "Audio/Source", "node.name": SOURCE, "node.always-process": True,
                             "priority.session": 0}), "-"], stdin=pcm,
            )
            self.addCleanup(audio.stop, source)

            def nodes():
                graph = json.loads(subprocess.check_output(["pw-dump"], timeout=2))
                return {props["node.name"]: props for node in graph
                        if "node.name" in (props := node.get("info", {}).get("props", {}))}

            fixture = wait_for(lambda: nodes().get(SOURCE))
            with patch.object(Settings, "save"), patch.object(Settings, "load", return_value=Settings(cleanup=False)):
                with patch.object(audio, "devices", side_effect=lambda: (
                        [{"name": SOURCE, "label": SOURCE, "serial": str(fixture["object.serial"])}], devices()[1])):
                    service = engine.Engine()
                    try:
                        wait_for(lambda: service.snapshot()["registered"])
                        self.assertFalse(np.any(record(.35)))
                        service.update({"mode": "direct", "gate": -80})
                        wait_for(lambda: service.snapshot()["output_level"] > .05)
                        result = record(.7)
                        self.assertGreater(audio.level(result), .09)
                        hz = np.argmax(abs(np.fft.rfft(result))) / .7
                        self.assertAlmostEqual(hz, 220, delta=3)

                        sink_name = "voicify_integration_preview"
                        module = subprocess.check_output(
                            ["pactl", "load-module", "module-null-sink", f"sink_name={sink_name}",
                             "sink_properties=priority.session=0"], text=True,
                        ).strip()
                        try:
                            sink = wait_for(lambda: nodes().get(sink_name))
                            capture = []
                            stream = audio.stream

                            def preview_stream(recording, *args, **kwargs):
                                if not recording:
                                    self.assertIsNotNone(capture[0].poll())
                                    self.assertTrue(capture[0].stdout.closed)
                                process = stream(recording, *args, **kwargs)
                                if recording:
                                    capture.append(process)
                                return process

                            with patch("voicify.control.request", return_value=service.snapshot()), \
                                    patch.object(audio, "headphone_sink", return_value=str(sink["object.serial"])), \
                                    patch.object(audio, "stream", side_effect=preview_stream):
                                audio.preview()
                        finally:
                            subprocess.run(["pactl", "unload-module", module], check=True)

                        virtual = nodes()[audio.MIC]
                        owner = virtual.get("application.process.id")
                        if owner is None:
                            graph = json.loads(subprocess.check_output(["pw-dump"], timeout=2))
                            client = next(item for item in graph if item["id"] == virtual["client.id"])
                            owner = client["info"]["props"]["application.process.id"]
                        os.kill(int(owner), signal.SIGTERM)
                        wait_for(lambda: nodes().get(audio.MIC, {}).get("object.serial") not in (None, virtual["object.serial"]))
                        wait_for(lambda: service.snapshot()["output_level"] > .05)
                        self.assertGreater(audio.level(record(.4)), .07)

                        service.update({"mode": "mute"})
                        wait_for(lambda: service.snapshot()["state"] == "Muted")
                        time.sleep(.25)
                        self.assertFalse(np.any(record(.35)))

                        demo = os.getenv("VOICIFY_TEST_MODEL")
                        if demo:
                            service.update({"mode": "convert", "model": demo, "backend": "CPU", "profile": "Smooth"})
                            wait_for(lambda: service.snapshot()["output_level"] > .003, 30)
                            converted = record(3)
                            self.assertGreater(audio.level(converted), .003)
                            self.assertFalse(np.allclose(converted[:len(tone)], tone, atol=.05))
                            print("Neural stream:", service.snapshot()["processing_ms"],
                                  "ms, dropped frames:", service.snapshot()["drops"], flush=True)
                    finally:
                        service.closing.set()
                        service.thread.join(5)
        wait_for(lambda: not devices()[1])
        self.assertEqual(subprocess.check_output(["pactl", "get-default-source"]), default_mic)
        self.assertEqual(subprocess.check_output(["pactl", "get-default-sink"]), default_sink)


if __name__ == "__main__":
    unittest.main()
