#!/bin/bash
# Run the unit test suite (the same command CI runs).
# Usage: bash scripts/run_tests.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/ -v
