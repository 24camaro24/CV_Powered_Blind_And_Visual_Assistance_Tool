#!/bin/bash

# Open Flask backend in default browser
echo "Opening frontend in browser..."
echo ""
echo "Make sure Flask backend is running! Use start_server.sh first"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_PATH="$SCRIPT_DIR/frontend/index.html"

# Try to open with default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "$FRONTEND_PATH"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "$FRONTEND_PATH"
else
    # Windows (Git Bash)
    start "$FRONTEND_PATH"
fi

echo "Frontend opened at: file://$FRONTEND_PATH"
echo "Backend should be running on: http://localhost:5000"
