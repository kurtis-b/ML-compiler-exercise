#!/usr/bin/env bash

if [[ -n "${MLIR_TUTORIAL_PIPELINE_COMMON_SH:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
MLIR_TUTORIAL_PIPELINE_COMMON_SH=1

pipeline_die() {
  echo "error: $*" >&2
  exit 1
}

pipeline_note() {
  echo "[mlir-tutorial] $*" >&2
}

pipeline_require_file() {
  local path="$1"
  [[ -f "$path" ]] || pipeline_die "required file not found: $path"
}

pipeline_require_dir() {
  local path="$1"
  [[ -d "$path" ]] || pipeline_die "required directory not found: $path"
}

pipeline_require_command() {
  local tool="$1"
  command -v "$tool" >/dev/null 2>&1 || pipeline_die "required command not found on PATH: $tool"
}

pipeline_has_library() {
  local dir="$1"
  local stem="$2"
  compgen -G "${dir}/${stem}.*" >/dev/null
}

pipeline_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "$(command -v python3)"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "$(command -v python)"
    return 0
  fi
  pipeline_die "python was not found. Set PYTHON, install python3, or activate the repo virtualenv."
}

PIPELINE_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
PIPELINE_REPO_ROOT="$(cd -- "${PIPELINE_HELPER_DIR}/.." && pwd)"

: "${BUILD_DIR:=${PIPELINE_REPO_ROOT}/build-ninja}"
: "${TORCH_MLIR_SOURCE_DIR:=${PIPELINE_REPO_ROOT}/externals/torch-mlir}"
: "${TORCH_MLIR_BUILD_DIR:=${PIPELINE_REPO_ROOT}/externals/torch-mlir/build}"
: "${OPENBLAS_DIR:=${PIPELINE_REPO_ROOT}/openblas}"
: "${MLIR_RUNNER_LIB_DIR:=${TORCH_MLIR_BUILD_DIR}/lib}"
: "${OPENBLAS_LIB_DIR:=}"
: "${HF_HUB_DISABLE_PROGRESS_BARS:=1}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${TRANSFORMERS_NO_ADVISORY_WARNINGS:=1}"
: "${TRANSFORMERS_VERBOSITY:=error}"

if [[ -z "${PYTHONWARNINGS:-}" ]]; then
  PYTHONWARNINGS="ignore::FutureWarning"
fi

export HF_HUB_DISABLE_PROGRESS_BARS
export TOKENIZERS_PARALLELISM
export TRANSFORMERS_NO_ADVISORY_WARNINGS
export TRANSFORMERS_VERBOSITY
export PYTHONWARNINGS

export PATH="${BUILD_DIR}/tools:${TORCH_MLIR_BUILD_DIR}/bin:${PATH}"

TUTORIAL_OPT="${TUTORIAL_OPT:-${BUILD_DIR}/tools/tutorial-opt}"
MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt 2>/dev/null || true)}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-$(command -v mlir-translate 2>/dev/null || true)}"
LLC="${LLC:-$(command -v llc 2>/dev/null || true)}"
TORCH_MLIR_OPT="${TORCH_MLIR_OPT:-$(command -v torch-mlir-opt 2>/dev/null || true)}"

if [[ -z "$MLIR_OPT" ]]; then
  MLIR_OPT="${TORCH_MLIR_BUILD_DIR}/bin/mlir-opt"
fi
if [[ -z "$MLIR_TRANSLATE" ]]; then
  MLIR_TRANSLATE="${TORCH_MLIR_BUILD_DIR}/bin/mlir-translate"
fi
if [[ -z "$LLC" ]]; then
  LLC="${TORCH_MLIR_BUILD_DIR}/bin/llc"
fi
if [[ -z "$TORCH_MLIR_OPT" ]]; then
  TORCH_MLIR_OPT="${TORCH_MLIR_BUILD_DIR}/bin/torch-mlir-opt"
fi

pipeline_activate_torch_mlir_python() {
  local python_bin
  python_bin="$(pipeline_python)"
  local mlir_python_dir="${TORCH_MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core"
  local torch_mlir_python_dir="${TORCH_MLIR_BUILD_DIR}/tools/torch-mlir/python_packages/torch_mlir"
  local fx_importer_dir="${TORCH_MLIR_SOURCE_DIR}/test/python/fx_importer"

  pipeline_require_file "$python_bin"
  pipeline_require_dir "$mlir_python_dir"
  pipeline_require_dir "$torch_mlir_python_dir"
  pipeline_require_dir "$fx_importer_dir"

  export PYTHONPATH="${mlir_python_dir}:${torch_mlir_python_dir}:${fx_importer_dir}${PYTHONPATH:+:${PYTHONPATH}}"
}

pipeline_require_common_tools() {
  pipeline_require_file "$TUTORIAL_OPT"
  pipeline_require_file "$MLIR_TRANSLATE"
  pipeline_require_file "$LLC"
}

pipeline_require_torch_mlir_tools() {
  pipeline_require_common_tools
  pipeline_require_file "$TORCH_MLIR_OPT"
}

pipeline_require_openblas() {
  local lib_dir="${OPENBLAS_LIB_DIR}"

  if [[ -z "$lib_dir" ]] && [[ -d "${OPENBLAS_DIR}/lib" ]] && pipeline_has_library "${OPENBLAS_DIR}/lib" "libopenblas"; then
    lib_dir="${OPENBLAS_DIR}/lib"
  fi

  if [[ -z "$lib_dir" ]] && [[ -d "/usr/lib/x86_64-linux-gnu" ]] && pipeline_has_library "/usr/lib/x86_64-linux-gnu" "libopenblas"; then
    lib_dir="/usr/lib/x86_64-linux-gnu"
  fi

  if [[ -z "$lib_dir" ]] && command -v ldconfig >/dev/null 2>&1; then
    local openblas_path=""
    openblas_path="$(ldconfig -p 2>/dev/null | awk '/libopenblas\.so/{print $NF; exit}')"
    if [[ -n "$openblas_path" ]]; then
      lib_dir="$(dirname "$openblas_path")"
    fi
  fi

  if [[ -z "$lib_dir" ]]; then
    pipeline_die "OpenBLAS library not found. Set OPENBLAS_DIR or OPENBLAS_LIB_DIR to your OpenBLAS install."
  fi

  if pipeline_has_library "$lib_dir" "libopenblas"; then
    OPENBLAS_LIB_DIR="$lib_dir"
    export OPENBLAS_LIB_DIR
    return 0
  fi
  pipeline_die "OpenBLAS library not found under ${lib_dir}. Set OPENBLAS_DIR or OPENBLAS_LIB_DIR to your OpenBLAS install."
}

pipeline_pybind11_includes() {
  local python_bin
  python_bin="$(pipeline_python)"
  "$python_bin" -m pybind11 --includes 2>/dev/null || pipeline_die "pybind11 is required in the active Python environment."
}

pipeline_python_ext_suffix() {
  local python_bin
  python_bin="$(pipeline_python)"
  "$python_bin" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX") or "")
PY
}

pipeline_resolve_cuda() {
  local nvcc_candidate=""

  if [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]]; then
    nvcc_candidate="${CUDACXX}"
  elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    nvcc_candidate="${CUDA_HOME}/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then
    nvcc_candidate="$(command -v nvcc)"
  elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    nvcc_candidate="/usr/local/cuda/bin/nvcc"
  fi

  if [[ -z "$nvcc_candidate" ]]; then
    pipeline_die "CUDA toolkit not found. Set CUDA_HOME or CUDACXX before running GPU pipelines."
  fi

  CUDACXX="$nvcc_candidate"
  if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(cd -- "$(dirname -- "${CUDACXX}")/.." && pwd)"
  fi

  if [[ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]]; then
    CUDA_LIB_DIR="${CUDA_HOME}/targets/x86_64-linux/lib"
  elif [[ -d "${CUDA_HOME}/lib64" ]]; then
    CUDA_LIB_DIR="${CUDA_HOME}/lib64"
  else
    pipeline_die "could not locate CUDA runtime libraries under ${CUDA_HOME}"
  fi

  if [[ -d /usr/lib/wsl/lib ]]; then
    CUDA_DRIVER_LIB_DIR="/usr/lib/wsl/lib"
  else
    CUDA_DRIVER_LIB_DIR="${CUDA_LIB_DIR}"
  fi

  local nvidia_smi="${NVIDIA_SMI:-}"
  if [[ -z "$nvidia_smi" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia_smi="$(command -v nvidia-smi)"
  elif [[ -z "$nvidia_smi" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    nvidia_smi="/usr/lib/wsl/lib/nvidia-smi"
  fi

  if [[ -z "${MLIR_CUDA_ARCH:-}" && -n "$nvidia_smi" ]]; then
    MLIR_CUDA_ARCH="$("$nvidia_smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')"
  fi

  if [[ -z "${MLIR_CUDA_ARCH:-}" ]]; then
    pipeline_die "unable to detect the NVIDIA compute capability. Set MLIR_CUDA_ARCH (for example 86 or 90)."
  fi

  export CUDACXX CUDA_HOME CUDA_LIB_DIR CUDA_DRIVER_LIB_DIR MLIR_CUDA_ARCH
}

pipeline_require_cuda_tools() {
  pipeline_require_common_tools
  pipeline_require_file "$MLIR_OPT"
  pipeline_resolve_cuda
  if ! pipeline_has_library "$MLIR_RUNNER_LIB_DIR" "libmlir_cuda_runtime"; then
    pipeline_die "torch-mlir was not built with CUDA runtime support under ${MLIR_RUNNER_LIB_DIR}. Rebuild torch-mlir with MLIR_ENABLE_CUDA_RUNNER=ON and MLIR_ENABLE_NVPTXCOMPILER=ON."
  fi
}
