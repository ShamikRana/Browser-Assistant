from huggingface_hub import snapshot_download

# Download Phi-3.5 mini ONNX model
snapshot_download(repo_id="microsoft/Phi-3.5-mini-instruct-onnx", cache_dir=r".\models\phi_models")

# Download small embedding model
snapshot_download(repo_id="thenlper/gte-small", cache_dir=r".\models\embedding_models")