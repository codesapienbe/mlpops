#!/bin/bash

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync
uv pip install -e .

uv run python -m crawl4ai-setup
uv run python -m playwright install --with-deps chromium
