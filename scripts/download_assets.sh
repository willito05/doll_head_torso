#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG À REMPLIR =========
# Lien DIRECT vers ton best.pt (GitHub Release recommandé)
WEIGHTS_URL="https://github.com/willito05/doll_head_torso/releases/download/v0.1.0/best.pt"

# (Optionnel) Lien vers un ZIP du dataset (ou laisse vide)
DATASET_ZIP_URL="https://github.com/willito05/doll_head_torso/releases/download/v0.1.1/Dataset_full_v01.zip"
# Nom du dossier dataset après extraction (adapter si besoin)
DATASET_DIR_NAME="Dataset"

UNZIP_TARGET="."
# ====================================

mkdir -p weights datasets

# ---- Weights ----
if [ -f weights/best.pt ]; then
  echo "[OK] weights/best.pt déjà présent (skip)"
else
  if [ -z "$WEIGHTS_URL" ]; then
    echo "[ERREUR] WEIGHTS_URL est vide. Mets un lien direct vers best.pt dans le script."
    exit 1
  fi
  echo "[DL] Téléchargement des poids -> weights/best.pt"
  curl -fL "$WEIGHTS_URL" -o weights/best.pt
  echo "[OK] weights/best.pt téléchargé."
fi

# ---- Dataset (optionnel) ----
if [ -n "$DATASET_ZIP_URL" ]; then
  if [ -d "datasets/$DATASET_DIR_NAME" ]; then
    echo "[OK] Dataset datasets/$DATASET_DIR_NAME déjà présent (skip)"
  else
    echo "[DL] Téléchargement du dataset..."
    tmpzip="/tmp/dataset_$$.zip"
    curl -fL "$DATASET_ZIP_URL" -o "$tmpzip"
    echo "[UNZIP] Extraction vers datasets/"
    mkdir -p datasets
    unzip -o "$tmpzip" -d datasets
    rm -f "$tmpzip"
    echo "[OK] Dataset extrait."
  fi
else
  echo "[INFO] Pas d’URL dataset (DATASET_ZIP_URL vide) : on saute l’étape dataset."
fi

# ---- Résumé ----
echo "------------------------------------------"
[ -f weights/best.pt ] && echo "[OK] weights/best.pt prêt"
[ -d "datasets/$DATASET_DIR_NAME" ] && echo "[OK] datasets/$DATASET_DIR_NAME prêt" || echo "[INFO] Aucun dataset local (optionnel)"
echo "Terminé."
