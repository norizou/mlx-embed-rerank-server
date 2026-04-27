#!/bin/bash
# Ruri Embed/Rerank server startup script

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Move to the project directory
cd "$SCRIPT_DIR"

echo "Starting Ruri Embed/Rerank server with uv..."

# Run the server with uvicorn using uv
# exec replaces the shell process with the command, ensuring signals are passed correctly
exec /opt/homebrew/bin/uv run --cache-dir /Users/norihito/AI/.uv_cache uvicorn ruri_embed_rerank_server:app --host 127.0.0.1 --port 1235
