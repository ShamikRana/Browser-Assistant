"""
Download the Phi-4 mini ONNX model (CPU and/or GPU variant) plus the small
embedding model used for the local RAG pipeline.

Usage:
    python download_model.py --device cpu
    python download_model.py --device gpu
    python download_model.py --device both
    python download_model.py --device cpu --check-only   # exit 1 if missing
"""

import argparse
import sys

import config

PHI4_REPO = "microsoft/Phi-4-mini-instruct-onnx"
EMBEDDING_REPO = "thenlper/gte-small"


def _phi4_present(device: str) -> bool:
    return (config.model_path_for(device) / "genai_config.json").is_file()


def _embeddings_present() -> bool:
    return (config.EMBEDDING_DIR / "config.json").is_file()


def download_phi4(device: str) -> None:
    # Imported lazily so --check-only works before dependencies are installed.
    from huggingface_hub import snapshot_download

    subfolder = config.GPU_SUBFOLDER if device == "gpu" else config.CPU_SUBFOLDER
    print(f"Downloading Phi-4 mini ONNX ({device}) -> {config.PHI4_DIR / subfolder}")
    snapshot_download(
        repo_id=PHI4_REPO,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(config.PHI4_DIR),
    )


def download_embeddings() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading embedding model -> {config.EMBEDDING_DIR}")
    snapshot_download(repo_id=EMBEDDING_REPO, local_dir=str(config.EMBEDDING_DIR))


def _targets(device: str) -> list[str]:
    if device == "both":
        return ["cpu", "gpu"]
    if device == "auto":
        # Auto only fetches the CPU build, which runs everywhere. Ask for
        # "gpu" or "both" explicitly to pull the CUDA variant.
        return ["gpu"] if _phi4_present("gpu") else ["cpu"]
    return [device]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download models for Browser Assistant (Phi-4)")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "both"],
        default="auto",
        help="Which Phi-4 ONNX variant to download (default: auto)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report whether models are present; exit 1 if anything is missing.",
    )
    args = parser.parse_args()

    devices = _targets(args.device)

    if args.check_only:
        missing = [d for d in devices if not _phi4_present(d)]
        if not _embeddings_present():
            missing.append("embeddings")
        if missing:
            print(f"Missing model files: {', '.join(missing)}")
            return 1
        print("All required models are present.")
        return 0

    for device in devices:
        if _phi4_present(device):
            print(f"Phi-4 ({device}) already downloaded, skipping.")
        else:
            download_phi4(device)

    if _embeddings_present():
        print("Embedding model already downloaded, skipping.")
    else:
        download_embeddings()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
