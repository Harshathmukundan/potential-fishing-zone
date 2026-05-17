#!/bin/bash
# Download models and data from Google Drive
echo "Starting Google Drive asset download..."
python download_assets.py

# Start the Flask app using Gunicorn
echo "Starting web server..."
gunicorn app:app --bind 0.0.0.0:$PORT
