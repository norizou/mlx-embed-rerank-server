# Launchdによる自動起動設定のまとめ

macOSのログイン時に `ruri_embed_rerank_server.py` が自動的に起動するように設定しました。

## 構成要素

### 1. 起動用シェルスクリプト
- **パス**: `/Users/norihito/AI/embed_reranker/run_ruri_server.sh`
- **役割**: 仮想環境(`.venv`)の有効化、カレントディレクトリの移動、および `uvicorn` によるサーバー起動（ポート: 1235）を行います。

### 2. Launchd設定ファイル (plist)
- **パス**: `~/Library/LaunchAgents/com.norihito.embed-reranker.plist`
- **役割**: システムログイン時に上記シェルスクリプトを実行し、プロセスの死活監視（KeepAlive）を行います。
- **ログ出力先**:
  - 標準出力: `~/Library/Logs/com.norihito.embed-reranker.log`
  - 標準エラー: `~/Library/Logs/com.norihito.embed-reranker.error.log`

## 管理コマンド

### サービスのステータス確認
```bash
launchctl list | grep embed-reranker
```

### サービスの手動停止
```bash
launchctl unload ~/Library/LaunchAgents/com.norihito.embed-reranker.plist
```

### サービスの手動開始
```bash
launchctl load ~/Library/LaunchAgents/com.norihito.embed-reranker.plist
```

### ログの確認 (リアルタイム)
```bash
tail -f ~/Library/Logs/com.norihito.embed-reranker.log
```
