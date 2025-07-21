#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPU_CPU_PROFILE="cpu"
exec $SCRIPT_DIR/run-compose.sh