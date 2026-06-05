# MLX対応検討

- GGUFからMLX対応のモデルへ移行
- Pythonのバージョン検討（3.13系）Pytorchの縛りから抜けたい。
- MLXの日本語対応モデルの確認
- MLXでAPIを提供する方法の検討

## MLX対応モデル選定

### Embeddingモデル

| モデル名 | 特徴 | MLXリポジトリパス (Hugging Face) |
| :--- | :--- | :--- |
| BGE-M3 (MLX) | 日本語を含む多言語対応の定番。FP16版が公開されており、Macでの精度と速度のバランスが良い。 | `mlx-community/bge-m3-mlx-fp16` |
| Jina Embeddings v3/v4 | 最新の日本語LLM（Qwen2.5/Qwen3ベース）をバックボーンに使用。検索性能が非常に高い。 | `mlx-community/jina-embeddings-v3` |
| Qwen3-VL-Embedding | 画像とテキストの両方を扱える最新モデル。日本語の文脈理解が極めて強力。 | `mlx-community/Qwen3-VL-Embedding-2B` |

### Rerankerモデル

| モデル名 | 特徴 | MLXリポジトリパス (Hugging Face) |
| :--- | :--- | :--- |
| Japanese Reranker v2 | BGE-M3ベースの日本語特化型。MLXで動かすことで大量の候補（100件以上）も一瞬で再ランク可能。 | `mlx-community/japanese-bge-reranker-v2-m3` |
| Qwen3-Reranker-0.6B | 非常に軽量ながら、Qwen3の強力な言語理解を引き継いでおり、低レイテンシで高精度。 | `mlx-community/Qwen3-Reranker-0.6B-4bit` |

## モデル選定

Qwen3-vl-2bのEmbed、Rerankerを使うのが良さそう。

## API提供方法の検討

### mlx-pythonのAPIサーバー

mlx-pythonのAPIサーバーをそのまま使うのが一番手っ取り早い。
現行のruri-embed-reranker API

<http://localhost:1235>
を踏襲する形。
