from pathlib import Path

from .config import DATA

ASSETS = Path(__file__).resolve().parent.parent / "demo-assets"
ENCODER = "d44be88b55b324bb4334772d97147d46d50ea19a39b6fe8214806d1a4e679c99"
CUDA_ENCODER = "cbddd8fa9352b3128df6359e2e3da6be0f9072e768e51e3efc3004e5030ea97f"
VOICES = {
    "Howatto": "264b34656c917b42ffc9943af90379e2f173417426b831bff61f961442f6523e",
    "Sittori": "2b7fb3094cb3bd777842efb29312ef976ce1403032ec7d8d516268b9039782c0",
    "Kakkoii": "07c6c501653b7b148dee1106e365a1e4638b85536e69aecd67624390ec2424fc",
}


def install(name="Howatto"):
    from .models import add_voice

    folder = ASSETS if ASSETS.is_dir() else DATA / "demo-assets"
    encoder = folder / "contentvec.onnx"
    voice = folder / f"aiso-{name.lower()}.onnx"
    cuda_encoder = folder / "contentvec-cuda.onnx"
    if not all(path.is_file() for path in (encoder, cuda_encoder, voice)):
        raise ValueError("The local demo pack is missing. You can import your own ONNX voice instead.")
    return add_voice(
        f"AISO · {name}", encoder, voice, 40000, 0, f"AISO {name} by 鶴乃 · ちはや神社 · MIT",
        {"encoder": ENCODER, "voice": VOICES[name], "cuda_encoder": CUDA_ENCODER}, cuda_encoder,
    )
