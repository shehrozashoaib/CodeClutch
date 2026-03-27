#!/usr/bin/env bash
set -euo pipefail
USER_ENV_DIR="/workspace/.venvs/user"
if [ "$#" -lt 1 ]; then
  echo "Usage: /workspace/run_user_code.sh <script.py> [args ...]" >&2
  exit 1
fi
exec "/workspace/.venvs/user/bin/python" ""
