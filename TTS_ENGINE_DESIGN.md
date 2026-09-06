# TTS エンジン設計書 — Qwen3-TTS / Irodori の二エンジン構成

`POST /v1/audio/speech` は **2 つの別系統の TTS エンジン**を 1 つの OpenAI 互換
エンドポイントで提供しています。本書はその設計判断・実装・テスト戦略をまとめたものです。

- 対象コード: [`mlx_embed_rerank_server.py`](mlx_embed_rerank_server.py) の `AVAILABLE_AUDIO_MODELS` と `audio_speech()`
- 実測値: [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §7（Irodori v4.1 と Qwen3-TTS の比較）、§6（Qwen3-TTS の初期評価）
- テスト全体の設計: [tests/TEST_DESIGN.md](tests/TEST_DESIGN.md)
- 利用方法: [README_JA.md](README_JA.md) の「音声合成（TTS）」

---

## 1. 背景と課題

このサーバーは当初 Qwen3-TTS のみを扱っていました。そこへ日本語特化の
**Irodori TTS**（Flow Matching / Rectified Flow DiT）を追加するにあたり、
次の問題が明らかになりました。

**2 つのエンジンは `generate()` の引数体系が根本的に異なる。**

| | Qwen3-TTS | Irodori |
|:---|:---|:---|
| 生成方式 | 音声トークンの自己回帰デコード | 連続 latent 上の Flow Matching |
| クローン方式 | ICL（参照音声 + **書き起こし**） | 参照音声のみ（書き起こし不要） |
| 長さの制御 | `speed`（話速） | `seconds` / `duration_scale`（出力長） |
| 声質の指示 | なし（base 版） | `caption`（VoiceDesign） |

とくに危険なのが **`ref_text` と `caption` の混同**です。名前が似ているため
「参照音声に対応するテキスト」として同一視しがちですが、実際には別物です。

- `ref_text` … 参照音声の**書き起こし**。Qwen3-TTS の ICL に必須
- `caption` … 「落ち着いた女性の声で〜」のような**声質の記述**。Irodori の VoiceDesign 用

さらに `caption` は `config.dit.use_caption_condition` が真のモデルでしか使われず、
偽のモデルでは**例外もログもなく黙って捨てられます**。両者をマッピングすると
「エラーは出ないが指定が一切効かない」という最も気づきにくい不具合になります。

---

## 2. 設計方針

### 2.1 エンジンはモデル種別でディスパッチする

エンジンの判別は、モデル ID の文字列マッチではなく**レジストリの `type`** で行います。

```python
AVAILABLE_AUDIO_MODELS = {
    "qwen3-tts-0.6b-base-8bit": {"type": "tts",         ...},
    "irodori-tts-v4.1-small-8bit": {"type": "tts_irodori", "supports_caption": True, ...},
}

TTS_TYPES = {"tts", "tts_irodori"}   # /v1/audio/speech が受理する種別
```

`get_tts()` の型判定はこの `TTS_TYPES` 集合で行います。当初 `type.startswith("tts")`
としていましたが、種別名の接頭辞に意味を持たせると将来 `tts_` で始まる非 TTS 種別を
足したときに誤って通してしまうため、明示集合に変更しました。

### 2.2 パラメータはエンジンごとに分離し、「無視」と「拒否」を使い分ける

片方のエンジンにしかないパラメータを他方へ送った場合の扱いは、**一律にはしません**。

| 分類 | 対象 | 挙動 | 理由 |
|:---|:---|:---|:---|
| 無視（警告ログ） | Irodori への `voice` / `ref_text` / `lang_code` / `max_tokens` | 200 | OpenAI 互換クライアント（OpenWebUI 等）は `voice` を**無条件に送る**。400 にすると接続できなくなる |
| 拒否（400） | Qwen3-TTS への `seconds` / `duration_scale` / `num_steps` / `cfg_guidance_mode` / `max_ref_seconds` / `instruct` | 400 | 標準の OpenAI TTS API に存在しないパラメータなので、送ってくるのは意図的な指定のみ。黙殺すると意図が達成されない |
| 拒否（400） | Qwen3-TTS への配列 `ref_audio` | 400 | 複数クリップは Irodori v4 固有の機能 |
| 拒否（400） | caption 条件を持たない Irodori 版への `instruct` | 400 | ライブラリ側が黙って捨てるため、サーバーで止めないと無言の失敗になる |

判断基準は「**そのパラメータを黙って捨てると、呼び出し側の意図が達成されないか**」です。
`voice` は Irodori では意味を持たず捨てても実害がない一方、`instruct` を捨てると
「声質を指示したのに反映されない」という結果になります。

### 2.3 変換できるものは変換する

`speed` は OpenAI TTS API の標準パラメータで、クライアントが送ってくる可能性が高い一方、
Irodori には対応する引数がありません。無視するのではなく `duration_scale` に変換します。

```python
# speed は「大きいほど速い」、duration_scale は「大きいほど長い」で逆向き
generate_kwargs["duration_scale"] = 1.0 / req.speed
```

`duration_scale` が明示された場合はそちらを優先します。

### 2.4 検証はモデルロードより前に行う

パラメータ検証は `manager.get_tts()` を呼ぶ**前**に完了させます。
Irodori / Qwen3-TTS の重みは数 GB あり、不正なリクエスト 1 本で
巨大なロードが走るのは受け入れられないためです。

```python
# 1. モデル名 → 2. 種別 → 3. エンジン別パラメータ  ここまで全て 400 判定
tts_model = manager.get_tts(model_name)   # 4. ここで初めてロード
```

この契約は `tests/test_errors.py::TestTTSEngineParameters` で固定しており、
同クラスに `audio` マーカーを付けていないこと自体が「重みを引かない」ことの表明です。
もしどれかがロードを誘発すれば、`-m "not audio"` の実行時間が数分単位で跳ね上がって露見します。

### 2.5 モデル固有の値はレジストリか実物の config から取る

「このモデルは caption を使えるか」「duration predictor を持つか」といった判定に
モデル ID の文字列マッチを使うと、モデル追加のたびに条件式が壊れます。

- **caption 対応の有無** → レジストリの `supports_caption` フラグ
- **duration predictor の有無** → ロード済みモデルの `config.dit.use_duration_predictor`

後者は実物の config を読むため、どのバージョンを登録しても自動的に正しく働きます。

---

## 3. 実装

### 3.1 リクエストモデル

`SpeechReq` はフィールドをエンジン別に区分してあります。

```python
class SpeechReq(BaseModel):
    input: str
    model: Optional[str] = DEFAULT_TTS
    response_format: Optional[str] = "mp3"
    # --- 共通 ---
    speed: Optional[float] = 1.0
    ref_audio: Optional[Union[str, List[str]]] = None
    # --- Qwen3-TTS 専用 ---
    voice / ref_text / lang_code / max_tokens
    # --- Irodori 専用 ---
    instruct / seconds / duration_scale / num_steps / cfg_guidance_mode / max_ref_seconds
```

`ref_audio` のみ両エンジン共通ですが、**Irodori v4 だけが配列を受け付けます**。
配列を渡すと各クリップが個別にエンコードされて連結されます（学習時の形式に一致）。

### 3.2 処理の流れ

```
POST /v1/audio/speech
  │
  ├─ モデル名は登録済みか            → 400 "Unsupported TTS model: ..."
  ├─ 種別は TTS_TYPES に含まれるか   → 400 "Model ... is not a TTS model"
  ├─ エンジン別パラメータの整合性     → 400（§2.2 の表）
  │
  ├─ manager.get_tts()   ← ここで初めて重みをロード（以降は常駐）
  │
  ├─ tts_irodori 分岐                  ├─ tts 分岐
  │   ref_audio / caption              │   voice / speed / ref_audio / ref_text
  │   seconds / duration_scale         │   lang_code
  │   num_steps / cfg_guidance_mode    │   max_tokens（ICL なら既定 8192）
  │   max_ref_seconds                  │
  │   + 無視パラメータの警告ログ        │
  │   + duration predictor 非搭載の警告 │
  │
  └─ generate() の全チャンクを連結 → audio_write → StreamingResponse
```

### 3.3 Qwen3-TTS の `max_tokens` 自動設定

ICL（`ref_audio` + `ref_text`）のときだけ `max_tokens` の既定を 8192 に引き上げます。
根拠は **ICL 経路がテキストを分割しない**ことです。

- 非 ICL 経路: `split_pattern`（既定 `"\n"`）でセグメントに分割し、`max_tokens` は**セグメントあたり**に適用される
- ICL 経路: 分割せず入力全体を 1 回の生成で処理するため、mlx-audio 既定の 4096 では長文が切れる

`lang_code` の指定有無とは無関係です（言語 ID は codec の**プレフィル**に 1 トークン載るだけで、
生成ステップ数 `for step in range(max_tokens)` には影響しません）。

### 3.4 Irodori の出力長制御

Irodori には話速の概念がなく、**出力長そのもの**を制御します。

| 指定 | 効果 |
|:---|:---|
| `seconds` | 出力長を秒で明示（`min_seconds` / `max_seconds` でクランプ） |
| `duration_scale` | duration predictor の推定長に対する倍率 |
| いずれも未指定 | duration predictor が推定（v4.1 は搭載） |

この方式の副次的な効果として、**Irodori の出力長は完全に決定論的**になります。
duration predictor が長さを決めてから Flow Matching で生成するため、
同じ入力なら 10 回とも同一長になります（[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) §7 で std=0.000）。
Qwen3-TTS は `temperature=0.9` のサンプリングが長さに直結するため std≈0.55 でばらつきます。
字幕同期のように尺が要件となる用途では、この差がエンジン選択の決め手になります。

`num_steps`（Euler ステップ数、既定 40）を下げても**長さは変わりません**。
ステップ数は積分の細かさ＝品質側のパラメータで、長さは duration predictor が決めるためです。
`num_steps=10` で生成時間は 33% 短縮します（3.960s → 2.645s）。

**`sequence_length` は `generate()` 経由では制御できません。** ライブラリ側で
kwargs のマージ前に `sampler_cfg.pop("sequence_length")` されるためです。
duration predictor を持たないモデル（v2 等）を登録した場合、`seconds` 未指定だと
`config.sampler.sequence_length`（750 フレーム = 30 秒、約 24GB）にフォールバックします。
サーバーはこれを検知して警告ログを出します。

### 3.5 モデル構成の変遷

| 版 | 構成 | 登録 |
|:---|:---|:---|
| v2 | クローンのみ / duration predictor なし | 削除 |
| v3 | base（クローン + 自動長さ）と VoiceDesign（caption）が**別モデル** | 削除 |
| **v4.1-Small** | **単一チェックポイントで全部**（クローン + caption + 自動長さ） | 8bit / fp16 |

v4 で 2 つのスクラッチ学習エンコーダが事前学習済みの **ModernBERT-ja-310m** バックボーンに
置き換わりました。このため `mlx-audio` 0.4.4 の `IrodoriDiTConfig` は
`text_encoder_type` / `pretrained_projector_*` を認識できず、`from_dict` がそれらを捨てて
旧アーキテクチャで構築するため重みロードに失敗します。依存の下限を **0.5.1** に上げているのはこのためです。

v4.1 が VoiceDesign を内包したことで `voice_design` フラグ（＝VoiceDesign 専用版の意）は
意味を失い、能力を表す **`supports_caption`** に改称しました。`instruct` の 400 判定は
このフラグを読み続けるため、将来 caption 条件のない版を再登録すればガードが自動で復活します。

---

## 4. テスト戦略

方針の詳細は [tests/TEST_DESIGN.md](tests/TEST_DESIGN.md) にあります。TTS 固有の要点は次の 3 つです。

### 4.1 「ロード前に弾く」ことをテスト構造で表現する

エンジン別パラメータの 400 は `tests/test_errors.py::TestTTSEngineParameters` に置き、
**`audio` マーカーを付けていません**。もしどれかがモデルロードを誘発すれば
実行時間が跳ね上がるため、マーカーの不在自体が §2.4 の契約のセンサーとして働きます。

### 4.2 参照音声はテスト側で合成する

`ref_audio` は**サーバーホスト上のパス**を指す仕様（アップロード未実装）のため、
テストは正弦波の WAV を `tmp_path` に書き出して渡します。リポジトリに音声素材を
コミットせずに `ref_audio` の経路全体を検証できます。

### 4.3 検証するのは契約であって音質ではない

生成音声の内容（声の類似度、話速、`seconds` どおりの長さ）は主観評価か音響解析が必要なため、
自動テストでは **HTTP 200 と RIFF ヘッダ**までを契約としています。

| テスト | 固定している契約 |
|:---|:---|
| `test_voice_clone_returns_wav` | `ref_audio` のみでクローンでき、書き起こしを要さない |
| `test_speed_maps_to_duration_scale` | `speed` は 400 にならず `duration_scale` へ変換される |
| `test_qwen3_params_are_ignored_not_rejected` | `voice` / `ref_text` は警告のみで 200 |
| `test_caption_only_voice_design` | `instruct` だけ（参照音声なし）で生成できる |
| `test_ref_audio_and_instruct_combine` | 統合チェックポイントは両方同時に使える |
| `test_multi_clip_ref_audio` | `ref_audio` の配列を受理する |
| `test_irodori_only_param_rejected_by_qwen3` | Irodori 専用パラメータは Qwen3 で 400 |
| `test_multi_clip_ref_audio_rejected_by_qwen3` | 配列は Qwen3 で 400 |

### 4.4 現状テストできない分岐

「caption 条件を持たない Irodori 版への `instruct` は 400」という分岐はサーバーに残していますが、
登録中の v4.1 は caption を持つため**到達不能**です。v2 / v3 base 等を再登録した時点で有効になります。
意図的に残している旨は TEST_DESIGN.md にも明記しています。

---

## 5. 既知の制約

- **`ref_audio` はサーバーホスト上のパス**。クライアントからのアップロードは未実装です。
- **音声モデルは自動アンロードされません**。Qwen3-VL のようなアイドルアンロード対象外で、
  一度ロードすると常駐します。fp16 と 8bit を両方叩くと両方が常駐します。
- **ストリーミング非対応**。Irodori 側は `stream=True` が `NotImplementedError` を投げるため、
  サーバーは全チャンクを収集してから 1 レスポンスで返します。
- **`DEFAULT_TTS` は Qwen3 のまま**（`qwen3-tts-0.6b-base-8bit`）。既存クライアントへの
  影響を避けるためで、Irodori へ切り替えるかはベンチマーク後の判断です。
