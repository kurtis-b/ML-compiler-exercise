#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../pipeline_common.sh"
cd "$PIPELINE_SCRIPT_DIR"

bash lower.sh
bash compile.sh

"${PIPELINE_SCRIPT_DIR}/build/vector_add_runner" "${GPU_TUTORIAL_SIZE}"
