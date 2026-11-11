from pathlib import Path

from .config import DATA

ASSETS = Path(__file__).resolve().parent.parent / "demo-assets"
ENCODER = "d44be88b55b324bb4334772d97147d46d50ea19a39b6fe8214806d1a4e679c99"
CUDA_ENCODER = "cbddd8fa9352b3128df6359e2e3da6be0f9072e768e51e3efc3004e5030ea97f"
VOICES = {
    "Howatto": "264b34656c917b42ffc9943af90379e2f173417426b831bff61f961442f6523e",
    "Sittori": "2b7fb3094cb3bd777842efb29312ef976ce1403032ec7d8d516268b9039782c0",
    "Kakkoii": "07c6c501653b7b148dee1106e365a1e4638b85536e69aecd67624390ec2424fc",
    "Sarasara": "8156ccb55f9c8e34fa64f5327bcad1b6de397bf771ebbc6c13db4fb980d27181",
    "Syakitto": "769a15e1976be984f24c90a1ba157a8e8f1aa6d9a3e45dd17260ab88d346003a",
    "p226": "4babd8fab3b45f133d370d8c9118111cf03e9c7fa45cb3ed2e72d9cacf993c7a",
    "p274": "d258b0b19bb4978e1b755e639bc6ce81d9077e93131210bbce6a528914648462",
}


def voice_file(name):
    return f"{'vctk' if name.startswith('p') else 'aiso'}-{name.lower()}.onnx"


def available():
    return [name for name in VOICES if asset(voice_file(name)).is_file()]


def asset(filename):
    local = ASSETS / filename
    installed = DATA / "demo-assets" / filename
    return installed if installed.is_file() else local


def install(name="Howatto"):
    from .models import add_voice

    encoder = asset("contentvec.onnx")
    voice = asset(voice_file(name))
    cuda_encoder = asset("contentvec-cuda.onnx")
    if not all(path.is_file() for path in (encoder, cuda_encoder, voice)):
        raise ValueError(
            "Download the optional voice pack from https://github.com/aronyok-io/Voicify/releases/latest "
            "and extract it into the Voicify data directory."
        )
    family, credit = ("VCTK", "Nekochu · Apache 2.0") if name.startswith("p") else (
        "AISO", "鶴乃 · ちはや神社 · MIT"
    )
    return add_voice(
        f"{family} · {name}", encoder, voice, 40000, 0, f"{family} {name} by {credit}",
        {"encoder": ENCODER, "voice": VOICES[name], "cuda_encoder": CUDA_ENCODER}, cuda_encoder,
    )
