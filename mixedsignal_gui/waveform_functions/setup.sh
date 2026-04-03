#!/usr/bin/env bash
#Setup for macOS/Linux
set -e
PY=python3

if [ ! -d "venv" ]; then
  $PY -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "To run the GUI: source venv/bin/activate && python -m mixedsignal_gui.main"
