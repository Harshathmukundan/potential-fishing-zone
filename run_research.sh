#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  run_research.sh — Full Research Pipeline
#  Trains ConvLSTM, then runs cross-validation for both models
# ──────────────────────────────────────────────────────────────
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT/backend/uploads/DATA"
VENV="$ROOT/venv"
MODEL_DIR="$ROOT/model"
OUTPUT="$ROOT/backend/saved_models"

echo ""
echo "🔬  PFZ Research Pipeline — Full Training + Cross-Validation"
echo "══════════════════════════════════════════════════════════════"

# Activate venv
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
  echo "✓ Virtual environment activated"
else
  echo "⚠  No venv found, using system Python"
fi

cd "$MODEL_DIR"

# ── Step 1: Train ConvLSTM (U-Net already has a checkpoint) ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1/4: Training ConvLSTM Model (5 epochs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python train.py --data_dir "$DATA_DIR" --skip_unet --epochs 5 --output_dir "$OUTPUT"

# ── Step 2: U-Net Temporal Block CV ──────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2/4: U-Net Temporal Block Cross-Validation (3-fold)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python cross_validate.py --data_dir "$DATA_DIR" --method temporal_block --k 3 --epochs 5 --model unet --output_dir "$OUTPUT"

# ── Step 3: ConvLSTM Walk-Forward CV ─────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3/4: ConvLSTM Walk-Forward Cross-Validation (3-fold)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python cross_validate.py --data_dir "$DATA_DIR" --method walk_forward --k 3 --epochs 5 --model convlstm --output_dir "$OUTPUT"

# ── Step 4: Generate updated training history ─────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4/4: Updating training history"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python generate_history.py

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✅ RESEARCH PIPELINE COMPLETE"
echo ""
echo "  Output files in: $OUTPUT"
echo "    • best_unet.keras          — U-Net model"
echo "    • best_convlstm.keras      — ConvLSTM model"
echo "    • cv_results_unet_temporal_block.json"
echo "    • cv_results_convlstm_walk_forward.json"
echo "    • cv_unet_temporal_block.png   — U-Net CV figure"
echo "    • cv_convlstm_walk_forward.png — ConvLSTM CV figure"
echo "    • training_history.json"
echo "══════════════════════════════════════════════════════════════"
