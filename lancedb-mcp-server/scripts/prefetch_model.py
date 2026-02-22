"""Pre-download the embedding model so it is baked into the Docker image.

Run during `docker build` to avoid downloading the model on first startup.
The model files are cached under $HF_HOME (default: ~/.cache/huggingface).
"""

import os

from sentence_transformers import SentenceTransformer

model_name = os.environ.get("LANCEDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
print(f"Pre-fetching embedding model: {model_name}")
model = SentenceTransformer(model_name)
# Run a dummy encode to ensure all weights are loaded and cached.
model.encode(["hello world"])
print(f"Model {model_name} cached successfully at {model.model_card_data.model_id}")
