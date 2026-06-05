from mlx_embeddings import load, generate
import numpy as np
import mlx.core as mx

print("Loading Embedding model...")
embed_model, embed_processor = load("mlx-community/bge-m3-mlx-fp16")
print("Model loaded successfully.")

texts = ["テスト文字列", "もう一つのテスト"]
print("Generating embeddings...")
output = generate(embed_model, embed_processor, texts=texts)
print("Generation successful. Processing output...")

if hasattr(output, "last_hidden_state"):
    print("Found last_hidden_state")
    state = np.array(output.last_hidden_state)
    cls_embeddings = state[:, 0, :]
    norm = np.linalg.norm(cls_embeddings, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-10, norm)
    embeddings = cls_embeddings / norm
    print("Embeddings shape:", np.array(embeddings).shape)
    print("Done!")
else:
    print("No last_hidden_state")
