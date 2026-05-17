#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
#  PFZ Navigator — Quick Start Script
#  Starts the backend Flask API and opens the frontend.
# ──────────────────────────────────────────────────────────────
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend/index.html"
VENV="$ROOT/venv"

echo ""
echo "🎣  PFZ Navigator — AI Geospatial Decision Support System"
echo "──────────────────────────────────────────────────────────"

# ── Check virtual environment ─────────────────────────────────
if [ -d "$VENV" ]; then
  echo "✓ Activating virtual environment..."
  source "$VENV/bin/activate"
else
  echo "⚠  No venv found. Using system Python."
fi

# ── Check required models ─────────────────────────────────────
MODEL_DIR="$BACKEND/saved_models"
echo ""
if [ -f "$MODEL_DIR/best_unet.keras" ]; then
  SIZE=$(du -h "$MODEL_DIR/best_unet.keras" | cut -f1)
  echo "✓ U-Net model found ($SIZE)"
else
  echo "✗ U-Net model NOT found — run training first:"
  echo "  cd model && python train.py --data_dir ../backend/uploads/DATA"
fi

if [ -f "$MODEL_DIR/best_convlstm.keras" ]; then
  SIZE=$(du -h "$MODEL_DIR/best_convlstm.keras" | cut -f1)
  echo "✓ ConvLSTM model found ($SIZE)"
else
  echo "✗ ConvLSTM model NOT found — run training first:"
  echo "  cd model && python train.py --data_dir ../backend/uploads/DATA --skip_unet"
fi

if [ -f "$MODEL_DIR/training_history.json" ]; then
  echo "✓ Training history found"
else
  echo "✗ Training history NOT found — generating..."
  (cd "$ROOT/model" && python generate_history.py)
fi

# ── Check data ────────────────────────────────────────────────
DATA_DIR="$BACKEND/uploads/DATA"
if [ -d "$DATA_DIR" ] && [ "$(ls -1 "$DATA_DIR"/*.nc 2>/dev/null | wc -l)" -gt 0 ]; then
  COUNT=$(ls -1 "$DATA_DIR"/*.nc | wc -l)
  echo "✓ $COUNT NetCDF dataset(s) found in uploads/DATA/"
else
  echo "⚠  No NetCDF files found. Upload data via the web UI."
fi

# ── Start backend ─────────────────────────────────────────────
echo ""
echo "🚀 Starting backend on http://localhost:5001 ..."
echo "──────────────────────────────────────────────────────────"

# Kill any existing instance on port 5001
lsof -ti:5001 2>/dev/null | xargs kill -9 2>/dev/null || true

(cd "$BACKEND" && python app.py --port 5001) &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# ── Open frontend ─────────────────────────────────────────────
echo ""
echo "🌐 Opening frontend..."
if command -v open &>/dev/null; then
  open "$FRONTEND"
elif command -v xdg-open &>/dev/null; then
  xdg-open "$FRONTEND"
else
  echo "   Open in browser: file://$FRONTEND"
fi

echo ""
echo "✅ PFZ Navigator is running!"
echo "   Backend API : http://localhost:5001/api/status"
echo "   Frontend    : file://$FRONTEND"
echo ""
echo "   Press Ctrl+C to stop the server."
echo ""

# Trap Ctrl+C to clean up
trap "kill $BACKEND_PID 2>/dev/null; echo '🛑 Server stopped.'; exit 0" INT TERM

# Keep script running
wait $BACKEND_PID
