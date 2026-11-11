import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import MODES
from .control import request


def start():
    state = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state")) / "voicify"
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(state / "engine.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab") as log:
        subprocess.Popen([sys.executable, "-m", "voicify", "serve"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    print(f"Service starting. Log: {state / 'engine.log'}")


def main():
    parser = argparse.ArgumentParser(prog="voicify")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "serve", "stop", "status", "preview", "compare", "diagnostics"):
        commands.add_parser(command)
    mode = commands.add_parser("mode")
    mode.add_argument("value", choices=MODES)
    model = commands.add_parser("add-voice")
    model.add_argument("name")
    model.add_argument("encoder", type=Path)
    model.add_argument("voice", type=Path)
    model.add_argument("--rate", type=int, required=True)
    model.add_argument("--speaker", type=int, default=0)
    model.add_argument("--credit", default="")
    model.add_argument("--encoder-sha")
    model.add_argument("--voice-sha")
    commands.add_parser("demo-voice")
    commands.add_parser("demo-voices")
    args = parser.parse_args()
    try:
        if args.command == "serve":
            from .service import serve
            serve()
        elif args.command == "start":
            start()
        elif args.command == "add-voice":
            from .models import add_voice
            identifier = add_voice(args.name, args.encoder, args.voice, args.rate, args.speaker, args.credit,
                                   {"encoder": args.encoder_sha, "voice": args.voice_sha})
            print(identifier)
        elif args.command in ("demo-voice", "demo-voices"):
            from .demo import available, install
            names = available() if args.command == "demo-voices" else ["Howatto"]
            for name in names or ["Howatto"]:
                print(install(name))
        elif args.command in ("preview", "compare"):
            from .audio import preview
            preview(compare=args.command == "compare")
        elif args.command == "diagnostics":
            from .audio import virtual_mic_listeners
            state = request("status")
            fields = ("state", "registered", "input_device", "cleanup_active", "backend", "buffer_ms",
                      "processing_ms", "capture_gaps", "inference_overruns", "output_underruns",
                      "output_backpressure", "fault")
            report = {key: state.get(key) for key in fields}
            report["virtual_mic_listeners"] = virtual_mic_listeners()
            print(json.dumps(report, indent=2))
        elif args.command == "mode":
            print(json.dumps(request("update", settings={"mode": args.value}), indent=2))
        else:
            print(json.dumps(request(args.command), indent=2))
    except Exception as exc:
        parser.exit(1, f"voicify: {exc}\n")


if __name__ == "__main__":
    main()
