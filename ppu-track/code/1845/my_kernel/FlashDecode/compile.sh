#!/usr/bin/env bash
set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_BIN=${PY_BIN:-/root/miniconda3/envs/aicas26/bin/python3}
TARGET_DIR=${TARGET_DIR:-/root/aicas26-lhd/submit_emls/my_kernel/flashDecode}
SUBMIT_ROOT=${SUBMIT_ROOT:-/root/aicas26-lhd/submit_emls}
RUNTIME_SRC=${RUNTIME_SRC:-"${THIS_DIR}/flashdecode_runtime.py"}

echo "[compile] FlashDecode root: ${THIS_DIR}"
echo "[compile] Python: ${PY_BIN}"
echo "[compile] Target: ${TARGET_DIR}"

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[compile][error] python not found: ${PY_BIN}" >&2
  exit 1
fi

if [[ ! -f "${THIS_DIR}/setup.py" ]]; then
  echo "[compile][error] missing setup.py in ${THIS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RUNTIME_SRC}" ]]; then
  echo "[compile][error] missing runtime source: ${RUNTIME_SRC}" >&2
  exit 1
fi

pushd "${THIS_DIR}" >/dev/null
"${PY_BIN}" setup.py build_ext --inplace
popd >/dev/null

SO_SRC=$(ls -1t "${THIS_DIR}"/flashdecode_ext*.so 2>/dev/null | head -n 1 || true)
if [[ -z "${SO_SRC}" || ! -f "${SO_SRC}" ]]; then
  echo "[compile][error] build completed but flashdecode_ext*.so not found" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
install -m 755 "${SO_SRC}" "${TARGET_DIR}/flashdecode_ext.so"
install -m 644 "${RUNTIME_SRC}" "${TARGET_DIR}/flashdecode_runtime.py"

echo "[compile] Deployed files:"
ls -lh "${TARGET_DIR}/flashdecode_ext.so" "${TARGET_DIR}/flashdecode_runtime.py"

echo "[compile] Running import self-check..."
"${PY_BIN}" - <<PY
import sys
from pathlib import Path

submit_root = Path("${SUBMIT_ROOT}").resolve()
sys.path.insert(0, str(submit_root))

mod = __import__("my_kernel.flashDecode.flashdecode_runtime", fromlist=["patch_qwen3vl_flashdecode"])
assert hasattr(mod, "patch_qwen3vl_flashdecode")
assert hasattr(mod, "get_flashdecode_stats")
print("[compile] Import check passed:", mod.__file__)
PY

echo "[compile] Done."
