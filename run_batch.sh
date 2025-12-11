#!/bin/bash
# Simple batch runner for JD -> recommendations

# Lấy thư mục gốc dự án
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Đường dẫn file JD CSV
CSV_FILE="$PROJECT_DIR/data/jds.csv"

# Đường dẫn output JSON
OUT_FILE="$PROJECT_DIR/outputs/batch_results.json"

# Python environment (tùy bạn)
PYTHON_BIN="python"  
# Hoặc dùng conda env nếu muốn:
# PYTHON_BIN="/Users/nhitruong/Documents/conda_install/envs/dm/bin/python"

echo "[RUN] Batch processing CSV..."
$PYTHON_BIN "$PROJECT_DIR/src/run_batch_recommend.py" \
    --csv "$CSV_FILE" \
    --n 10 \
    --top-k 20 \
    --out "$OUT_FILE"

echo "[DONE] Saved results to: $OUT_FILE"
