from mlx_embeddings import load, generate
import sys

print("Loading model...")
try:
    model, processor = load("mlx-community/bge-m3-mlx-fp16")
    print("Model loaded.")
    out = generate(model, processor, texts=["hello world"])
    print(type(out))
    print(dir(out))
    if hasattr(out, 'last_hidden_state'):
        print("last_hidden_state shape:", out.last_hidden_state.shape)
    if hasattr(out, 'pooler_output'):
        print("pooler_output shape:", out.pooler_output.shape)
    if hasattr(out, 'embeddings'):
        print("embeddings shape:", out.embeddings.shape)
    if hasattr(out, 'hidden_states'):
        print("hidden_states:", type(out.hidden_states))
except Exception as e:
    print("Error:", e)
