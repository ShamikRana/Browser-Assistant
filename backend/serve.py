"""Entrypoint for running the backend.

Used by the start scripts and by the auto-start service, which needs a plain
executable target with no shell prompts or interactive pauses.

Usage:
    python serve.py                 # auto-detect device
    python serve.py --device gpu
    python serve.py --log-file server.log
"""

import argparse
import logging
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Browser Assistant backend")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-file", default=None, help="Write logs here instead of stdout")
    args = parser.parse_args()

    if args.device:
        os.environ["DEVICE"] = args.device

    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = BACKEND_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        )

    import config  # imported after DEVICE is set so it picks up the override

    if not (config.model_path_for(config.DEVICE) / "genai_config.json").is_file():
        print(
            f"Phi-4 model not found for device '{config.DEVICE}'.\n"
            f"Run: python download_model.py --device {config.DEVICE}",
            file=sys.stderr,
        )
        return 1

    import uvicorn

    run_options = {}
    if args.log_file:
        # Logging is already configured to the file above; keep uvicorn from
        # re-attaching its own stdout handlers.
        run_options["log_config"] = None

    uvicorn.run(
        "server:app",
        host=args.host or config.HOST,
        port=args.port or config.PORT,
        **run_options,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
