#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source transformer-lens-env/bin/activate

HOST_IP=$(hostname -I | awk '{print $1}')

echo "[backend]  starting on http://${HOST_IP}:8000"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "[frontend] starting on http://${HOST_IP}:5173"
cd frontend && npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM

wait $BACKEND_PID $FRONTEND_PID
