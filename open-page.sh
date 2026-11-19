#!/bin/sh
# Start the seam carving page and open it in the browser. Ctrl+C stops it.
cd "$(dirname "$0")" || exit 1
exec python3 frontend/serve.py "$@"
