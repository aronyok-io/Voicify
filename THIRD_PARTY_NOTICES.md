# Third-party notices

## Local demo assets

- **AISO Howatto, Sittori and Kakkoii** were created by 鶴乃 and distributed by ちはや神社. The publisher licenses the weights under MIT. This build uses local ONNX conversions of the 40 kHz RVC v2 checkpoints. [Publisher and terms](https://booth.pm/ja/items/4701666)
- **ContentVec** is by the ContentVec authors and builds on HuBERT in fairseq. The local graph was adapted from Tigre Gótico’s export with MatMul-only weight quantization. [ContentVec](https://github.com/auspicious3000/contentvec) · [Fairseq](https://github.com/facebookresearch/fairseq) · [Export provenance and MIT notice](https://huggingface.co/TigreGotico/voiceclonnx-rvc/blob/cbfbabdabe6a1414292bade8f857fb6074abf130/PROVENANCE.md)
- **VCClient / w-okada voice-changer**, copyright 2022 Wataru Okada, MIT. The local AISO graphs were exported with the VCClient model architecture at the pinned revision. [VCClient license](https://github.com/w-okada/voice-changer/blob/d8ef15799470193f7c8176ef471245753a656626/LICENSE)
- **RVC-Project**, copyright 2023 RVC-Project, MIT. [Upstream license](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)

## Runtime

The Python runtime uses NumPy, SciPy, ONNX, ONNX Runtime, and their locked dependencies. Fedora supplies Qt for Python, PipeWire, WirePlumber, and the PulseAudio client utilities. NVIDIA runtime wheels are optional and are licensed separately. The installed distributions retain their license files.
