#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG (fill in if needed) =========
# Direct link to your best.pt hosted on a GitHub Release
WEIGHTS_URL="https://github.com/willito05/doll_head_torso/releases/download/v0.1.0/best.pt"

# Optional dataset ZIP (leave commented/empty to skip)
# DATASET_ZIP_URL="https://github.com/willito05/doll_head_torso/releases/download/v0.1.1/Dataset_full_v01.zip"
# DATASET_DIR_NAME="Dataset"      # root folder name inside the ZIP
# UNZIP_TARGET="."                # where to extract ('.' or 'datasets')
# =============================================

mkdir -p weights datasets

# ---- Weights (required) ----
if [ -f weights/best.pt ]; then
  echo "[OK] weights/best.pt already present (skip)"
else
  if [ -z "${WEIGHTS_URL:-}" ]; then
    echo "[ERROR] WEIGHTS_URL is empty. Please set a direct link to best.pt."
    exit 1
  fi
  echo "[DL] Downloading weights -> weights/best.pt"
  curl -fL "$WEIGHTS_URL" -o weights/best.pt
  echo "[OK] weights/best.pt downloaded."
fi

# ---- Dataset (optional) ----
if [ -n "${DATASET_ZIP_URL:-}" ]; then
  DATASET_DIR_NAME="${DATASET_DIR_NAME:-Dataset}"
  UNZIP_TARGET="${UNZIP_TARGET:-datasets}"

  if [ -d "$UNZIP_TARGET/$DATASET_DIR_NAME" ]; then
    echo "[OK] Dataset $UNZIP_TARGET/$DATASET_DIR_NAME already present (skip)"
  else
    echo "[DL] Downloading dataset ZIP ..."
    tmpzip="/tmp/dataset_$$.zip"
    curl -fL "$DATASET_ZIP_URL" -o "$tmpzip"
    echo "[UNZIP] Extracting to $UNZIP_TARGET/"
    mkdir -p "$UNZIP_TARGET"
    unzip -o "$tmpzip" -d "$UNZIP_TARGET"
    rm -f "$tmpzip"
    echo "[OK] Dataset extracted to $UNZIP_TARGET/$DATASET_DIR_NAME"
  fi
else
  echo "[INFO] No dataset URL set (skipping dataset)."
fi

# ---- Summary ----
echo "------------------------------------------"
[ -f weights/best.pt ] && echo "[OK] weights/best.pt ready"
if [ -n "${DATASET_ZIP_URL:-}" ] && [ -d "${UNZIP_TARGET:-datasets}/${DATASET_DIR_NAME:-Dataset}" ]; then
  echo "[OK] dataset ready at ${UNZIP_TARGET:-datasets}/${DATASET_DIR_NAME:-Dataset}"
else
  echo "[INFO] No local dataset (optional)"
fi
echo "Done."
