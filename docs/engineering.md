# Engine and verification

A physical PipeWire source is chosen by its stable name and captured using its current object serial. The stream is pinned to that serial so WirePlumber cannot move capture to another device when the mic disappears. Voicify feeds a dedicated PipeWire `Audio/Source` from its own PCM stream. There is no playback sink for other applications to send audio to. The application leaves the system input and output selections alone.

The engine owns capture, conversion, and the virtual microphone. Studio talks to it over a private Unix-domain socket. The engine discovers graph changes, recreates the virtual microphone after a connection loss, and resumes capture when the chosen input returns. Closing Studio leaves the engine active. A fresh start always begins muted.

The pipeline applies gain and a short noise gate to 20 ms PCM frames. RVC uses rolling context and smooths each block join over 3 ms. Only the current block enters the output. The model hears the mic before gating, so it keeps the context around quiet sounds. The gate shapes the converted output, including its attack and release. Studio preview records four seconds from the virtual microphone into memory, closes the recording stream, checks the headphone route, and plays the clip once. When inference falls behind, the input queue stays bounded and stale frames are discarded. Buffer profiles configure 80/120/180 ms blocks on CUDA and 180/240/360 ms blocks on CPU. These are processing blocks. The studio reports the measured inference time separately from the block size.

## Verification

Run `.venv/bin/python -m unittest discover -s tests` from a source checkout. Set `VOICIFY_TEST_PIPEWIRE=1` in a desktop session to include the virtual mic and preview test. It uses a synthetic source and checks that the original system audio defaults are restored.
