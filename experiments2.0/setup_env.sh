#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-gpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${TARGET}" != "gpu" && "${TARGET}" != "cpu" ]]; then
  echo "Usage: ./setup_env.sh [gpu|cpu]" >&2
  exit 2
fi

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  "${PYTHON_BIN}" -m venv "${PROJECT_ROOT}/.venv"
fi
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
"${PYTHON}" -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
if [[ "${TARGET}" == "gpu" ]]; then
  "${PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements-gpu-cu118.txt"
else
  "${PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements-cpu.txt"
fi
"${PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements-common.txt"
"${PYTHON}" -m pip install --no-deps -e "${PROJECT_ROOT}"
"${PYTHON}" -m pytest \
  "${PROJECT_ROOT}/tests" \
  "${PROJECT_ROOT}/RQ1/tests" \
  "${PROJECT_ROOT}/RQ2/tests" \
  "${PROJECT_ROOT}/RQ3/tests" \
  "${PROJECT_ROOT}/RQ4/tests" \
  "${PROJECT_ROOT}/RQ5/tests" \
  --basetemp "${PROJECT_ROOT}/.pytest-tmp" \
  -q
if [[ "${TARGET}" == "gpu" ]]; then
  "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_environment.py"
else
  "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_environment.py" --allow-cpu
fi
"${PYTHON}" -m pip freeze --exclude-editable > "${PROJECT_ROOT}/requirements-lock.txt"
