# Contributing to Voicify

On Fedora, install PipeWire, WirePlumber, PySide6 and uv. From the checkout, run:

```sh
./setup cpu
.venv/bin/python -m unittest discover -s tests
./launch
```

For audio changes, compare Direct and Convert using the same take. Try quiet speech, pauses, and louder words. In KRecorder, set **Settings → Audio Input** to **Voicify Virtual Microphone** and record a short clip.

When reporting audio issues, include your distro, Voicify version, voice model, and the output from **Audio check**. Review the copied text for device and app names before posting it. Share recordings only if you are comfortable making them public.

Keep pull requests focused and include the steps you used to check the change. For voice models, include the source, license, and permission to redistribute the weights.
