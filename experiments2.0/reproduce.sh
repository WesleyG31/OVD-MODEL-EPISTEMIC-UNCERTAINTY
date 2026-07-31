#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "mini" && "${MODE}" != "full" ]]; then
  echo "Usage: bash reproduce.sh [smoke|mini|full]" >&2
  exit 2
fi

bash "${PROJECT_ROOT}/setup_env.sh" gpu
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
"${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_data.py"
"${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_model.py"
"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_model.py"
"${PYTHON}" "${PROJECT_ROOT}/scripts/run_rq1.py" --mode "${MODE}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/run_rq2.py" --mode "${MODE}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/run_rq3.py" --mode "${MODE}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/run_rq4.py" --mode "${MODE}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/run_rq5.py" --mode "${MODE}"
