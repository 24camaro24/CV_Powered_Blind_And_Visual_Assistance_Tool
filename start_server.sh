#!/bin/bash

# Voice-Guided Navigation Assistant - Quick Start Script

echo "========================================="
echo "Voice-Guided Navigation Assistant"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Navigate to backend
cd "$(dirname "$0")/backend" || exit

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "========================================="
echo "Starting Flask Server..."
echo "========================================="
echo ""

python3 app.py
