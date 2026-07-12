#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"

cd "${ROOT}"
git submodule update --init third_party/LIBERO third_party/openpi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m pip install -r requirements/libero-eval.txt
python -m pip install --no-deps -e third_party/LIBERO
python -m pip install -e third_party/openpi/packages/openpi-client

python scripts/check_release_environment.py --scope install

echo "SafeLoop environment ready: ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
