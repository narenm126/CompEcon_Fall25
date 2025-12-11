#!/usr/bin/env bash
# finalproject.sh
# Reproduce all results for the ECON 833 Final Project.

set -euo pipefail



# -------------------------------------
# 0. Move to directory of script
# --------------------------------------
cd "$(dirname "$0")"

echo "=== ECON 833 Final Project: Reproducibility Script ==="

# --------------------------------------
# 1. Create / activate Python virtual environment
# --------------------------------------
if [ ! -d ".venv" ]; then
    echo "[1] Creating virtual environment (.venv)..."
    python -m venv .venv
fi

echo "[1] Activating virtual environment..."

# Windows (Git Bash) uses Scripts/, Linux/macOS use bin/
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install python deps (minimal)
pip install --quiet numpy scipy matplotlib

echo "[1] Virtual environment ready."

# --------------------------------------
# 2. Run Python scripts
# --------------------------------------
echo "[2] Running model solution script: solve_model.py"
python solve_model.py

echo "[2] Running figure generation script: make_figures.py"
python make_figures.py

echo "[2] Python analysis complete."

# --------------------------------------
# 3. Compile LaTeX PDF
# --------------------------------------
TEX_MAIN="FinalProject_Mahor.tex"

if [ -f "$TEX_MAIN" ]; then
    echo "[3] Compiling LaTeX: $TEX_MAIN"

    pdflatex -interaction=nonstopmode "$TEX_MAIN" >/dev/null
    pdflatex -interaction=nonstopmode "$TEX_MAIN" >/dev/null

    echo "[3] PDF generated: ${TEX_MAIN%.tex}.pdf"
else
    echo "[3] WARNING: $TEX_MAIN not found, skipping PDF compilation."
fi

echo "=== Reproduction finished successfully. ==="
