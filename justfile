#!/usr/bin/env -S just --justfile

@setup:
    @echo "Checking environment"
    uv lock --check
    uv pip install -e .

@test:
    @echo "Running pytest"
    uv run pytest -v

@docs:
  uv run python docs/parse.py
