# テスト設計書 — mlx-embed-rerank-server

本書は `tests/` 配下の自動テストの設計方針をまとめたものです。
**正本はコード**（`mlx_embed_rerank_server.py`）であり、テストは「実装が現に示す振る舞い」を固定して、
ドキュメント（`README.md` / `README_JA.md` / `GEMINI.md`）との乖離を検知することを目的とします。

## 1. スコープと方針

| 項目 | 方針 |
|:---|:---|
| テストレベル | **API 結合テスト**のみ。稼働中のサーバー（ポート 1235）へ実 HTTP リクエストを送る |
| モック | 使用しない。モデル推論も実行する（MLX の実挙動が検証対象のため） |
| 単体テスト | 現時点では対象外。サーバーは単一モジュールで、公開インターフェースが HTTP のみのため |
| 期待値の管理 | `tests/data/test_cases.json` に集約。テストコードにマジックナンバーを埋め込まない |
| 失敗の意味 | 「実装が変わった」か「実装とドキュメントが食い違った」かのどちらか。仕様変更時は JSON とドキュメントを同時に更新する |

### 前提条件

- サーバーが `http://localhost:1235` で稼働していること（未起動ならスイート全体を skip）
- ヘルスチェック専用サーバーが `http://localhost:1236` で稼働していること（未応答なら該当テストのみ skip）
- モデルの重みが Hugging Face キャッシュに存在すること（初回は自動ダウンロードのため長時間かかる）
- `qwen3-vl-*` のテストには `torch` / `torchvision` が必要（未導入なら自動 skip）

## 2. ファイル構成

```text
tests/
├── TEST_DESIGN.md      # 本書
├── conftest.py         # fixture・マーカー登録・共通ヘルパー
├── test_health.py      # /health（1235）と監視用プローブ（1236）
├── test_embeddings.py  # POST /v1/embeddings
├── test_rerank.py      # POST /v1/rerank・/rerank
├── test_audio.py       # POST /v1/audio/transcriptions・/v1/audio/speech
├── test_errors.py      # エラー応答とリクエストバリデーション
└── data/
    ├── test_cases.json # 全期待値（モデル一覧・次元数・スコア範囲・ステータスコード）
    └── asr_output.txt  # ベンチマーク時の ASR 出力（参考データ）
```

`conftest.py` が提供するもの:

| 名前 | 種別 | 用途 |
|:---|:---|:---|
| `client` | fixture (session) | 1235 への httpx クライアント。タイムアウト 600 秒（遅延ロード対応）。未起動時は skip |
| `probe_client` | fixture (session) | 1236 への httpx クライアント |
| `health` | fixture (session) | `/health` のレスポンス（1 回だけ取得して共有） |
| `test_cases` | fixture (session) | `test_cases.json` の内容 |
| `embed()` / `rerank()` | ヘルパー | 200 を確認しつつ JSON を返す |
| `l2_norm()` | ヘルパー | ベクトルのノルム計算 |
| `requires_vl` | マーカー | `torch`/`torchvision` 未導入時の skip 条件 |

## 3. マーカーと実行方法

| マーカー | 対象 | 既定 |
|:---|:---|:---|
| `vl` | Qwen3-VL（2B）を使うテスト | `torch`/`torchvision` があれば実行、無ければ自動 skip |
| `audio` | STT / TTS エンドポイント（モデルロードを伴い最も遅い） | 実行される。除外する場合は `-m "not audio"` |

```bash
uv sync --extra dev

uv run pytest tests/                    # 全件
uv run pytest tests/ -m "not audio"     # 音声を除く高速版
uv run pytest tests/ -m audio           # 音声のみ
uv run pytest tests/test_rerank.py -v   # ファイル単位
```

`--strict-markers` を有効にしているため、未登録のマーカーはエラーになります（追加時は `pyproject.toml` の `markers` にも登録）。

## 4. テストケース一覧

### 4.1 `test_health.py` — ヘルスチェック

| ケース | 検証内容 |
|:---|:---|
| `test_status_ok` | `status == "ok"` |
| `test_available_models_match_server_config` | `available_embed` / `available_rerank` / `available_audio` が `AVAILABLE_*_MODELS` と一致（**モデル追加時に最初に落ちるテスト**） |
| `test_loaded_model_lists_present` | `loaded_*_models` 4 種が list として存在 |
| `test_loaded_models_are_a_subset_of_available` | ロード済みモデルは必ず利用可能モデルの部分集合 |
| `test_defaults_are_available` | ドキュメント記載のデフォルト 4 種が実在 |
| `test_no_undocumented_fields` | `/health` のキー集合が完全一致（過去ドキュメントにあった `reranker_ready` のような幽霊フィールドの検知） |
| `test_probe_returns_ok` | 1236 が `{"status": "ok"}` を返す |
| `test_probe_answers_any_path` | プローブはパスを問わず 200（`run_mlx_server.sh` の URL 変更で 404 にならない保証） |

### 4.2 `test_embeddings.py` — 埋め込み

| ケース | 検証内容 |
|:---|:---|
| `test_shape_and_normalization`（5 モデル） | 件数・`index` の並び・次元数・**L2 ノルムが 1.0**（実装は全モデルで CLS を正規化） |
| `test_openai_compatible_envelope`（5 モデル） | `object` / `data` / `model` / `usage` のキー構造 |
| `test_default_model_when_model_omitted` | `model` 省略時に `DEFAULT_EMBED`（`bge-m3`）が使われる |
| `test_string_input_equals_single_element_list` | `input` の文字列指定と 1 要素配列指定が等価 |
| `test_deterministic` | 同一入力は同一ベクトル（キャッシュ／サンプリング揺らぎがないこと） |
| `test_gemma_prefix_changes_the_vector` | `gemma-3-300m` のみ `input_type` でプレフィックスが変わりベクトルが変化 |
| `test_standard_models_ignore_input_type` | 標準モデルは `input_type` の影響を受けない |
| `test_instruction_is_ignored_by_non_vl_models` | `instruction` は VL 以外で無視される（ドキュメントの明記事項） |
| `test_related_text_scores_higher_than_unrelated` | デフォルトモデルで「猫/犬」＞「猫/量子力学」の意味的健全性 |

対象モデルと期待次元: `gemma-3-300m` 768 / `bge-m3` 1024 / `bge-m3-8bit` 1024 / `qwen3-0.6b-embed` 1024 / `qwen3-vl-embedding-2b` 2048（`vl`）。

### 4.3 `test_rerank.py` — リランク

| ケース | 検証内容 |
|:---|:---|
| `test_ranking_and_response_shape`（2 モデル） | レスポンスのキー構造、`document` が除去されていること、`index` の妥当性・重複なし、スコア降順、期待される 1 位 |
| `test_score_range` | `qwen3-0.6b` のスコアが 0〜1（yes/no の softmax 確率） |
| `test_default_model_when_model_omitted` | `DEFAULT_RERANK`（`qwen3-0.6b`）が使われる |
| `test_alias_path_matches_v1_path` | `/rerank` と `/v1/rerank` が同一結果 |
| `test_top_k_limits_results` / `test_top_k_larger_than_documents` | 件数の切り詰め挙動 |
| `test_top_k_zero_returns_every_document` | `top_k=0` は全件返却（`if req.top_k:` の仕様） |
| `test_empty_documents_returns_empty_results` | 空配列でもエラーにならず `results: []` |
| `test_deterministic` | 同一入力で完全同一のレスポンス |
| `test_default_model_ignores_instruction` | `qwen3-0.6b` は `instruction` の有無でスコアが変わらない |

### 4.4 `test_audio.py` — 音声（`audio` マーカー）

| ケース | 検証内容 |
|:---|:---|
| `test_returns_text_field` | STT が `{"text": str}` のみを返す（正弦波なので内容は問わない） |
| `test_default_model_when_model_omitted`（STT） | `DEFAULT_ASR` で動作 |
| `test_temporary_file_is_cleaned_up` | `/tmp/stt_*` が処理後に残らない（`finally` での削除） |
| `test_returns_audio_bytes` | TTS の Content-Type と `Content-Disposition: filename=speech.<fmt>`、本文が非空 |
| `test_wav_output_has_riff_header` | wav 出力が実際に RIFF ヘッダを持つ |
| `test_default_model_when_model_omitted`（TTS） | `DEFAULT_TTS` で動作 |
| `TestSpeechIrodori::test_voice_clone_returns_wav` | Irodori は `ref_audio` のみでクローンでき、書き起こしを要さない |
| `TestSpeechIrodori::test_speed_maps_to_duration_scale` | Irodori に `speed` はないが、サーバーが `duration_scale` へ逆数変換するので 200 |
| `TestSpeechIrodori::test_qwen3_params_are_ignored_not_rejected` | `voice` / `ref_text` は警告ログのみで **400 にしない**（OpenAI 互換クライアント互換性） |
| `TestSpeechIrodori::test_voice_design_accepts_instruct` | VoiceDesign 版のみ `instruct`（caption）で声質を指定できる |
| `test_requested_audio_model_appears_in_health` | 音声モデルは自動アンロードされず `/health` に残り続ける |

### 4.5 `test_errors.py` — エラー応答

実装が意図的に**非対称**なので、その差をそのまま固定しています。

| ケース | 期待 |
|:---|:---|
| `test_embeddings_wraps_the_400_into_a_500` | `/v1/embeddings` は未知モデルでも **500**（catch-all が内部 400 を包む） |
| `test_rerank_returns_400` | `/v1/rerank` は **400** + `Unsupported rerank model: ...` |
| `test_stt_returns_400` / `test_tts_returns_400` | 音声は **400** + `Unsupported ASR/TTS model: ...` |
| `test_tts_model_rejected_by_stt_endpoint` ほか | 種別違いのモデル指定は **400** + `Model ... is not an ASR/TTS model` |
| `TestTTSEngineParameters::test_irodori_only_param_rejected_by_qwen3` | `seconds` 等の Irodori 専用パラメータを Qwen3-TTS に渡すと **400** |
| `TestTTSEngineParameters::test_instruct_rejected_by_irodori_base_model` | caption 条件を持たない base 版への `instruct` は **400**（黙って無視しない） |
| `TestTTSEngineParameters::test_voice_design_needs_instruct_or_ref_audio` | VoiceDesign 版は `instruct` か `ref_audio` のいずれか必須で、無指定は **400** |
| `TestRequestValidation` | 必須フィールド欠落は Pydantic により **422**、未定義ルートは **404** |

`TestTTSEngineParameters` は `manager.get_tts()` より**前**に検証される契約を固定しています。
不正リクエストで数 GB のモデルロードが走らないことが要件なので、これらは `audio` マークを付けません。

## 5. 意図的にテストしていないこと

| 対象 | 理由 |
|:---|:---|
| 自動フォールバック（30 秒無使用での Qwen3-VL アンロード） | 判定が 30 秒周期タイマーのため 1 ケースに最大 60 秒かかり、`torch` 必須でもある。手動確認（`/health` の `loaded_*` を監視）に委ねる |
| スーパーバイザーの再起動動作 | プロセスの kill を伴い、テスト実行環境（launchd 常駐）を壊す |
| Qwen3-TTS のボイスクローン（`ref_audio` + `ref_text`） | ICL エンコードが重く、参照音声の書き起こしが必要。Irodori 側（`TestSpeechIrodori`）で `ref_audio` の経路自体は押さえている |
| 生成音声の**内容**（声の類似度、話速、`seconds` どおりの長さ） | 主観評価か音響解析が必要。ここでは 200 と RIFF ヘッダまでを契約とする |
| 推論の絶対精度・速度 | `BENCHMARK_REPORT.md` の計測が担当。テストは順位・型・構造など安定した性質のみを見る |
| 画像入力 | HTTP API はテキスト専用（`input` は文字列／文字列配列のみ） |

## 6. メンテナンス指針

1. **モデルを追加・削除したら**: `tests/data/test_cases.json` の `expected_*_models` と、`embedding` / `rerank` セクションの期待値を更新。`test_embeddings.py` / `test_rerank.py` のパラメータ一覧にも追加する。
2. **エンドポイントの挙動を変えたら**: 対応するテストを先に更新し、`README.md` / `README_JA.md` / `GEMINI.md` の該当箇所も同時に直す（テストはドキュメントの裏付けとして書かれている）。
3. **新しいマーカーを足したら**: `pyproject.toml` の `markers` と `conftest.py` の `pytest_configure` の両方に登録する（`--strict-markers` のため）。
4. **テストが遅くなってきたら**: まず `-m "not audio"` で分離できるかを検討する。
