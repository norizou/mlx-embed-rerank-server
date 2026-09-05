# Launchdによる自動起動設定のまとめ

macOSのログイン時に `mlx_embed_rerank_server.py` が自動的に起動するように設定しました。

## 構成要素

### 1. 起動用シェルスクリプト
- **パス**: `/Users/norihito/Projects/AI/Workspace/embed_reranker/run_mlx_server.sh`（plist もこのパスを参照）
- **役割**: スクリプト自身の位置へ移動し、`uv` でサーバーを起動（API ポート: 1235）した上で、フォアグラウンドのスーパーバイザーとして常駐します。
- **機能**: `kill` / `stop`, `restart`, `status` オプションによるプロセス管理が可能です。
- **監視**: 30 秒ごとに **ヘルスチェック専用ポート 1236**（サーバー内の別デーモンスレッド）へ curl（タイムアウト 10 秒）。2 回連続失敗（約 60 秒無応答）でハングと判定し、ポート 1235 のプロセスを停止して再起動します。
  MLX 推論は uvicorn のイベントループをブロックするため、1235 ではなく 1236 を監視することで長時間の TTS/ASR 推論による誤再起動を防いでいます（経緯は `BENCHMARK_REPORT.md` §6.6）。

### 2. Launchd設定ファイル (plist)
- **パス**: `~/Library/LaunchAgents/com.norihito.embed-reranker.plist`
- **役割**: システムログイン時に上記シェルスクリプトを実行し、プロセスの死活監視（`RunAtLoad` + `KeepAlive`）を行います。
- **WorkingDirectory**: `/Users/norihito/Projects/AI/Workspace/embed_reranker`
- **環境変数**: `PATH` に `/opt/homebrew/bin` を含めており、`uv` の実行を保証しています。
- **ログ出力先**:
  - 標準出力: `~/Library/Logs/com.norihito.embed-reranker.log`
  - 標準エラー: `~/Library/Logs/com.norihito.embed-reranker.error.log`

## 管理コマンド

### サーバーの操作（推奨）
スクリプトに組み込まれた管理機能を使用します。
```bash
./run_mlx_server.sh status   # 稼働状況とロード済みモデルの確認（1235 の /health を参照）
./run_mlx_server.sh restart  # 再起動
./run_mlx_server.sh kill     # 停止（KeepAliveにより即座に再起動されます）
```

### 稼働確認

```bash
curl http://localhost:1235/health   # ロード済み／利用可能モデルを含む詳細
curl http://localhost:1236/health   # スーパーバイザーが監視している軽量プローブ
```

### サービス自体の停止（自動起動を無効化したい場合）
```bash
launchctl unload ~/Library/LaunchAgents/com.norihito.embed-reranker.plist
```

### サービス自体の開始
```bash
launchctl load ~/Library/LaunchAgents/com.norihito.embed-reranker.plist
```

### ログの確認 (リアルタイム)
```bash
tail -f ~/Library/Logs/com.norihito.embed-reranker.log
```
