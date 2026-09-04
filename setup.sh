#!/usr/bin/env bash
# Recreates every system-level dependency this dump needs. Idempotent; safe to re-run.
# Dump protocol: nothing may live only on the machine. If you install something, add it here.
set -euo pipefail

# --- JPEG XL tools: jxl_from_tree (tree -> .jxl), djxl (.jxl -> .png), jxlinfo ---------------
if [[ "$(uname)" == "Darwin" ]]; then
    # Homebrew's jpeg-xl ships jxl_from_tree. `reinstall` also repairs stale dylib links
    # (an old bottle was linked against a removed imath version and aborted on launch).
    brew reinstall jpeg-xl
else
    # Linux: distro packages usually omit jxl_from_tree, so build the tools from source.
    JXL_VERSION=v0.11.1
    sudo apt-get update
    sudo apt-get install -y cmake ninja-build clang git pkg-config libbrotli-dev libgif-dev \
        libjpeg-dev libpng-dev libwebp-dev
    if [[ ! -d /opt/libjxl ]]; then
        sudo git clone --depth 1 --branch "$JXL_VERSION" --recursive https://github.com/libjxl/libjxl.git /opt/libjxl
    fi
    cmake -S /opt/libjxl -B /opt/libjxl/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_EXAMPLES=OFF
    cmake --build /opt/libjxl/build --target jxl_from_tree djxl cjxl jxlinfo
    sudo ln -sf /opt/libjxl/build/tools/{jxl_from_tree,djxl,cjxl,jxlinfo} /usr/local/bin/
fi

# --- Python deps for art/build.py and art/experiments/ -----------------------------------------
PYTHON=/opt/homebrew/opt/python@3.10/bin/python3.10
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -m pip install --quiet fire numpy pillow

# Self-check: jxl_from_tree with no args prints usage and exits 1 (that is the healthy case).
out=$(jxl_from_tree 2>&1 || true)
[[ "$out" == *Usage* ]] && echo "setup OK: jxl_from_tree works" || { echo "setup FAILED: jxl_from_tree broken: $out"; exit 1; }
