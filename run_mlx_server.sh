#!/bin/bash
PORT=1235
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Move to the project directory
cd "$SCRIPT_DIR"

# Command definition
RUN_CMD="uv run uvicorn mlx_embed_rerank_server:app --host 0.0.0.0 --port $PORT"

# --- Monitoring Configuration ---
CHECK_INTERVAL=30    # 監視間隔 (秒)
MAX_FAILS=2          # 連続失敗の許容回数 (30秒 × 2回 = 約60秒ハングで判定)
CURL_TIMEOUT=10      # healthチェック自体のタイムアウト時間 (秒)

function stop_server() {
    PID=$(lsof -ti:$PORT)
    if [ -n "$PID" ]; then
        echo "Stopping server on port $PORT (PID: $PID)..."
        kill $PID
        # Wait for it to actually stop
        while lsof -ti:$PORT > /dev/null; do sleep 0.5; done
        echo "Stopped."
    else
        echo "No server found running on port $PORT."
    fi
}

function monitor_loop() {
    local fail_count=0
    echo "Starting health monitor (Interval: ${CHECK_INTERVAL}s, Max fails: ${MAX_FAILS})..."
    
    while true; do
        sleep $CHECK_INTERVAL
        
        # /health にアクセスしてHTTPステータスコードを取得
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time $CURL_TIMEOUT http://localhost:$PORT/health)
        
        if [ "$HTTP_CODE" = "200" ]; then
            if [ $fail_count -gt 0 ]; then
                echo "Server recovered from partial hang."
            fi
            fail_count=0
        else
            fail_count=$((fail_count + 1))
            echo "Health check failed ($fail_count/$MAX_FAILS). HTTP_CODE: ${HTTP_CODE:-Timeout}"
            
            if [ $fail_count -ge $MAX_FAILS ]; then
                echo "[$(date)] Server hung for $((CHECK_INTERVAL * MAX_FAILS))s! Restarting..."
                stop_server
                echo "Restarting MLX Server..."
                export UV_PYTHON=3.13
                $RUN_CMD &
                fail_count=0
                
                # 起動待ちのため少し待機
                sleep 5
            fi
        fi
    done
}

function start_server() {
    PID=$(lsof -ti:$PORT)
    if [ -n "$PID" ]; then
        echo "Server is already running on port $PORT (PID: $PID)."
    else
        echo "Starting MLX Embedding + Rerank Server with uv..."
        export UV_PYTHON=3.13
        
        # launchd や Ctrl+C で終了された時にバックグラウンドのサーバーも道連れにする
        trap 'echo "Shutting down supervisor..."; stop_server; exit 0' SIGINT SIGTERM
        
        $RUN_CMD &
        
        # モニターをフォアグラウンドループで動かし、スクリプトをスーパーバイザーとして常駐させる
        monitor_loop
    fi
}

case "$1" in
    kill|stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        PID=$(lsof -ti:$PORT)
        if [ -n "$PID" ]; then
            echo "Server is running (PID: $PID)."
            # Fetch loaded models from health endpoint
            HEALTH=$(curl -s --max-time 2 http://localhost:$PORT/health)
            if [ $? -eq 0 ]; then
                echo "$HEALTH" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'Loaded Embedding: {d.get(\"loaded_embed_models\", [])}'); print(f'Loaded Rerank: {d.get(\"loaded_rerank_models\", [])}')"
            else
                echo "Could not fetch health status (Server might be starting up)."
            fi
        else
            echo "Server is not running."
        fi
        ;;
    *)
        start_server
        ;;
esac
