#!/bin/bash
PORT=1235
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Move to the project directory
cd "$SCRIPT_DIR"

# Command definition
RUN_CMD="uv run uvicorn mlx_embed_rerank_server:app --host 0.0.0.0 --port $PORT"

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

function start_server() {
    PID=$(lsof -ti:$PORT)
    if [ -n "$PID" ]; then
        echo "Server is already running on port $PORT (PID: $PID)."
    else
        echo "Starting MLX Embedding + Rerank Server with uv..."
        export UV_PYTHON=3.13
        exec $RUN_CMD
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

