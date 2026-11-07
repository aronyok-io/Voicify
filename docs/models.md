# Voice graphs

Voicify accepts a standalone ONNX content encoder and a standalone RVC synthesizer. The encoder takes mono float PCM at 16 kHz and returns 50 Hz content features in `[1, T, 256]` or `[1, T, 768]`. Encoders may also take an integer or Boolean `attention_mask`.

The synthesizer must return mono PCM at 32, 40, or 48 kHz. The importer tests the graph pair with the selected speaker and sample rate before listing the voice. Supported signatures include the RVC project's `phone`/ `phone_lengths` exports and VCClient-style `feats`/ `p_len` exports, with or without F0 inputs. RVC v1 and v2 share this path through their feature width. The demo exercises v2 without F0. Retrieval indexes and separate formant controls are outside this build.

The importer verifies the graph structure, checks a SHA-256 digest when supplied, and limits each graph to 512 MiB. It rejects external tensor data, custom operator domains outside the ONNX and ONNX Runtime domains, symlinks at the final file path, and PyTorch checkpoints. Imported graphs are stored by SHA-256 and verified again on load. Use graphs from sources you trust and voice weights you have permission to use.

## AISO demo voices

Howatto, Sittori and Kakkoii are Japanese-trained RVC v2 voices at 40 kHz, speaker ID 0. The publisher describes it as a merged voice made from ten consenting contributors and licenses the model weights under MIT. The demo uses a restricted PyTorch weight load for the one-time export, verifies the source hash, and runs the exported graph through the same importer used by the studio. Voicify's runtime uses the exported graphs.

- [Publisher and original terms](https://booth.pm/ja/items/4701666)
- [Pinned checkpoint mirror](https://huggingface.co/wok000/vcclient_model/tree/c46962577db8cf3fba2b3fc3526af98eb6611840/rvc_v2_chihaya_jinja)
- [Pinned ContentVec export and provenance](https://huggingface.co/TigreGotico/voiceclonnx-rvc/tree/cbfbabdabe6a1414292bade8f857fb6074abf130)

The CPU ContentVec copy uses dynamic int8 quantization for MatMul weights only. The CUDA backend uses the original float graph so the quantized operators do not require CPU transfers. Convolution stays in float because the publisher’s full int8 export uses a convolution operator the tested CPU runtime could not execute.

| Artifact | SHA-256 |
| --- | --- |
| Original Howatto checkpoint | `8fc32652b8ec472615af3f6661a50a195dd54990e0f517bd0dbd66db59813146` |
| Original float ContentVec export | `cbddd8fa9352b3128df6359e2e3da6be0f9072e768e51e3efc3004e5030ea97f` |
| Local Howatto ONNX export | `264b34656c917b42ffc9943af90379e2f173417426b831bff61f961442f6523e` |
| Original Sittori checkpoint | `7696d020ced92d9d5502ca40943ee7d35120988a92b70909fa099fc4e24a8477` |
| Local Sittori ONNX export | `2b7fb3094cb3bd777842efb29312ef976ce1403032ec7d8d516268b9039782c0` |
| Original Kakkoii checkpoint | `ae33b60655a59ecb37135892ed4a95ba326b412d6589dea251fced95e18aef31` |
| Local Kakkoii ONNX export | `07c6c501653b7b148dee1106e365a1e4638b85536e69aecd67624390ec2424fc` |
| Local ContentVec ONNX export | `d44be88b55b324bb4334772d97147d46d50ea19a39b6fe8214806d1a4e679c99` |

The demo graphs are optional and installed separately. Put the five ONNX files in the Voicify data directory under `demo-assets` to import them from Studio.
