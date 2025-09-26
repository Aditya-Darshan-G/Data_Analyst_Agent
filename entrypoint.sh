#!/usr/bin/env bash
set -e
uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
