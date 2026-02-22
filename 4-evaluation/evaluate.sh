#!/bin/bash
set -e

echo "[SAGE-ARCH] Running Mechanistic Weight Norm Inspection..."
uv run python3 4-evaluation/inspect_norms.py

echo "[SAGE-ARCH] Visualizing Weight Norm Trajectories..."
uv run python3 4-evaluation/plot_norms.py

echo "[SAGE-ARCH] Generating Generalisation Gap Plot..."
uv run python3 4-evaluation/generalisation_gap_monitor.py

echo "[SAGE-ARCH] Running the Ashtavakra Audit..."
uv run python3 4-evaluation/ashtavakra_audit.py
