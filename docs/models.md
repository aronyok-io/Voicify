# Voice graphs

Voicify accepts a standalone ONNX content encoder and a standalone RVC synthesizer. The encoder takes mono float PCM at 16 kHz and returns 50 Hz content features in `[1, T, 256]` or `[1, T, 768]`. Encoders may also take an integer or Boolean `attention_mask`.

The synthesizer must return mono PCM at 32, 40, or 48 kHz. The importer tests the graph pair with the selected speaker and sample rate before listing the voice. Supported signatures include the RVC project's `phone`/ `phone_lengths` exports and VCClient-style `feats`/ `p_len` exports, with or without F0 inputs. RVC v1 and v2 share this path through their feature width. The demo exercises v2 without F0. Retrieval indexes and separate formant controls are outside this build.

The importer verifies the graph structure, checks a SHA-256 digest when supplied, and limits each graph to 512 MiB. It rejects external tensor data, custom operator domains outside the ONNX and ONNX Runtime domains, symlinks at the final file path, and PyTorch checkpoints. Imported graphs are stored by SHA-256 and verified again on load. Use graphs from sources you trust and voice weights you have permission to use.

## AISO demo voices

Howatto, Sittori, Kakkoii, Sarasara and Syakitto are Japanese-trained RVC v2 voices at 40 kHz, speaker ID 0. The publisher describes it as a merged voice made from ten consenting contributors and licenses the model weights under MIT. The demo uses a restricted PyTorch weight load for the one-time export, verifies the source hash, and runs the exported graph through the same importer used by the studio. Voicify's runtime uses the exported graphs.

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

The demo graphs are optional and installed separately. Put the ONNX files in the Voicify data directory under `demo-assets` to import them from Studio.

## VCTK male voices

Nekochu trained p226 and p274 on VCTK speech and released the checkpoints under Apache 2.0. These are English-trained, pitch-guided RVC v2 voices at 40 kHz, speaker ID 0. Voicify distributes ONNX conversions of the original checkpoints and uses the same ContentVec encoder as the AISO voices.

| Voice | Original checkpoint SHA-256 | ONNX SHA-256 |
| --- | --- | --- |
| AISO Sarasara | `ff8c3058dc6ef17f46bd178190cd38fdf46fc06f1d47c12fae55a0cc54398a4e` | `8156ccb55f9c8e34fa64f5327bcad1b6de397bf771ebbc6c13db4fb980d27181` |
| AISO Syakitto | `569257d45848dada4ba604dfb9a60496d3043eb112e82c81d71eb37be942be50` | `769a15e1976be984f24c90a1ba157a8e8f1aa6d9a3e45dd17260ab88d346003a` |
| VCTK p226 | `3c4c2e8ea977d7c08793126aace4baa47028b4c1bf7cd46ae5a9c359b0216693` | `4babd8fab3b45f133d370d8c9118111cf03e9c7fa45cb3ed2e72d9cacf993c7a` |
| VCTK p274 | `bcba880bf4d0a51598b69fe68b8c128ab95a9aae4c6d39add61dd4f847275c66` | `d258b0b19bb4978e1b755e639bc6ce81d9077e93131210bbce6a528914648462` |

[Model card](https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample) · [Pinned VCTK checkpoints](https://huggingface.co/Nekochu/RVC-VCTK_Voice-sample/tree/005c2f948ee9dafd7e3aa7f261b4c3a24beebeef/M) · [VCTK corpus](https://datashare.ed.ac.uk/handle/10283/3443)

To install the release pack, run this from your download folder, then choose **Add demo voices** in Studio.

```bash
data_dir="${XDG_DATA_HOME:-$HOME/.local/share}/voicify"
mkdir -p "$data_dir"
tar -xzf voicify-voices-1.2.0.tar.gz -C "$data_dir"
```
