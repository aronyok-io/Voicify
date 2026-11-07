# Voicify

**A local voice studio for Linux.**

Turn a physical mic into a single virtual microphone for Discord, OBS, browsers, and games. Use Direct for your regular voice, Convert for an RVC voice, or Mute to stop capture. The audio engine stays active when you close the studio.

## Install on Fedora

Install the RPM for Fedora x86-64, then open **Voicify** from your app menu.

```bash
sudo dnf install ./voicify-1.0.0-1.fc44.x86_64.rpm
```

The first launch asks you to choose CPU or NVIDIA CUDA and downloads the Python runtime into your user account. To change engines later, stop audio and run `voicify setup cpu` or `voicify setup cuda`.

Voices you imported before installing are still in your library. For a fresh setup, import compatible ONNX graphs or put the demo pack in `${XDG_DATA_HOME:-$HOME/.local/share}/voicify/demo-assets` and click **Add demo voices**. See [voice formats](docs/models.md).

To remove the app, stop audio and run `sudo dnf remove voicify`. Your models, settings and runtime stay in your home directory.

## Run from source

Voicify uses PipeWire, WirePlumber, Pulse compatibility, and the Fedora PySide6 package. The inference environment uses Python 3.12 through `uv`.

```bash
./setup cpu
./launch
```

Use `./setup cuda` for the NVIDIA runtime. Stop audio before switching runtimes. The CUDA setup installs the locked CUDA 12 and cuDNN 9 Python wheels into Voicify’s own environment. The host still needs a working NVIDIA driver. Use `./setup cpu` to switch back.

1. Click **Start audio**. Voicify starts muted.
2. Pick a physical microphone or keep **Automatic microphone**. Click **Direct** and check both level meters.
3. Choose **Voicify Virtual Microphone** as the input in the app you want to test. Use the app’s microphone test or record a short local clip in OBS.
4. Click **Add demo voices** if the local test pack is present. Pick **Howatto**, **Sittori**, or **Kakkoii**, then click **Convert**.
5. **Echo cancellation** uses your default playback device as the reference to clean the mic before conversion. **Extra noise filter** adds stronger background noise suppression when needed. The **CLEAN MIC** indicator shows when the filter is active.
6. Use **Mute** to stop capture and keep the virtual mic visible. **Stop audio** removes the device and ends the background process. With headphones connected, use **Record preview** to hear what the virtual mic sends. Speak while it records, then listen to the clip. The clip stays in memory and Studio discards it when playback ends.

If Auto cannot initialize CUDA, Voicify attempts CPU inference and displays the provider in use. If conversion fails, the virtual mic goes silent. **Direct** is an explicit choice.

## Voice library

Import standalone RVC and ContentVec/HuBERT ONNX graphs using **Import ONNX**. Choose the model’s native sample rate and speaker ID. F0-guided voices expose pitch. The three demo voices use pitch-free RVC v2 models.

[Model formats and provenance](docs/models.md) · [Architecture and review](docs/engineering.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Test and control

```bash
./launch status
./launch mode mute
./launch stop
.venv/bin/python -m unittest discover -s tests -v
```

For the live synthetic-audio test, stop Voicify first and set `VOICIFY_TEST_PIPEWIRE=1` before running the tests. To include neural output, install the demo with `./launch demo-voice` and pass its printed ID as `VOICIFY_TEST_MODEL`. The test uses its own tone source and leaves the physical microphone alone.

Settings and imported graphs are stored in the standard XDG config and data directories under `voicify`. The engine log is under `$XDG_STATE_HOME/voicify/engine.log` or `~/.local/state/voicify/engine.log`. The runtime socket is private to your Linux account.

## Build the Fedora RPM

From a committed checkout with `rpm-build` and `desktop-file-utils` installed:

```bash
rpm_tree="$(mktemp -d)"
mkdir -p "$rpm_tree/SOURCES"
git archive --format=tar.gz --prefix=voicify-1.0.0/ HEAD > "$rpm_tree/SOURCES/voicify-1.0.0.tar.gz"
rpmbuild --define "_topdir $rpm_tree" -bb packaging/voicify.spec
find "$rpm_tree/RPMS" -name '*.rpm'
```
