# Launchdによる自動起動設定のまとめ

macOSのログイン時に `mlx_embed_rerank_server.py` が自動的に起動するように設定しました。

## 構成要素

### 1. 起動用シェルスクリプト
- **パス**: `[プロジェクトの絶対パス]/run_mlx_server.sh` (例: `/Users/norihito/AI/embed_reranker/run_mlx_server.sh`)
- **役割**: カレントディレクトリの移動、および `uv` によるサーバー起動（ポート: 1235）を行います。
- **機能**: `kill`, `restart`, `status` オプションによるプロセス管理が可能です。

### 2. Launchd設定ファイル (plist)
- **パス**: `~/Library/LaunchAgents/com.norihito.embed-reranker.plist`
- **役割**: システムログイン時に上記シェルスクリプトを実行し、プロセスの死活監視（KeepAlive）を行います。
- **環境変数**: `PATH` に `/opt/homebrew/bin` を含めており、`uv` の実行を保証しています。
- **ログ出力先**:
  - 標準出力: `~/Library/Logs/com.norihito.embed-reranker.log`
  - 標準エラー: `~/Library/Logs/com.norihito.embed-reranker.error.log`

## 管理コマンド

### サーバーの操作（推奨）
スクリプトに組み込まれた管理機能を使用します。
```bash
./run_mlx_server.sh status   # 稼働状況とロード済みモデルの確認
./run_mlx_server.sh restart  # 再起動
./run_mlx_server.sh kill     # 停止（KeepAliveにより即座に再起動されます）
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
