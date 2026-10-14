"""Pre-download the embedding model so it is baked into the Docker image.

Run during `docker build` to avoid downloading the model on first startup.
The model files are cached under $HF_HOME (default: ~/.cache/huggingface).
At runtime the image sets HF_HUB_OFFLINE=1, so the server never contacts
Hugging Face and only the model downloaded here can be used.
"""

import os

from sentence_transformers import SentenceTransformer

model_name = os.environ.get("LANCEDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
print(f"Pre-fetching embedding model: {model_name}")
model = SentenceTransformer(model_name, trust_remote_code=False)
# Encode a mixed-language sample so all weights and the tokenizer are loaded.
vectors = model.encode([
    "hello world",
    "字幕不会写入磁盘",
    "面接の練習モード",
])
print(
    f"Model {model_name} cached: {vectors.shape[1]} dims, "
    f"max_seq_length={model.max_seq_length}"
)
