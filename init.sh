#!/bin/bash
git submodule update --init --recursive
rm -rf .venv
uv venv -p python3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ./transformers
