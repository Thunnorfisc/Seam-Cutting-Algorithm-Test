#!/bin/sh
# Install the Python packages this project needs (Linux / macOS): sh install.sh
cd "$(dirname "$0")" || exit 1

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python 3.10 or newer is needed: https://www.python.org/downloads/"
    exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "This project needs Python 3.10 or newer. This one is: $("$PY" --version 2>&1)"
    exit 1
fi

echo "Installing the packages in requirements.txt ..."
if ! "$PY" -m pip install -r requirements.txt; then
    echo
    echo "pip could not install the packages. If it says the environment is"
    echo "\"externally managed\", make a virtual environment and run this again inside it:"
    echo "    $PY -m venv .venv && . .venv/bin/activate && sh install.sh"
    exit 1
fi

echo
exec "$PY" tools/check_setup.py
