#!/bin/bash
# fix-venv.sh - Fix venv for Kubernetes pod

set -e  # Exit on any error

VENV_PATH="/workspace/writeable/repos/open-r1_safety/openr1"
REQUIRED_PYTHON="python3.11"

echo "=== Setting up Python venv for batch job ==="

# Step 1: Install Python 3.11 if not present
if ! command -v $REQUIRED_PYTHON &> /dev/null; then
    echo "Python 3.11 not found. Installing..."
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-dev python3.11-venv > /dev/null
    echo "Python 3.11 installed successfully"
else
    echo "Python 3.11 already available"
fi

# Step 2: Verify venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Step 3: Fix symlinks
echo "Fixing venv symlinks..."
cd "$VENV_PATH/bin"
rm -f python python3 python3.11
ln -s $(which $REQUIRED_PYTHON) python3.11
ln -s python3.11 python3
ln -s python3.11 python

# Step 4: Verify it works
source "$VENV_PATH/bin/activate"
ACTUAL_VERSION=$(python --version 2>&1)
echo "Activated venv - $ACTUAL_VERSION"

# Step 5: Verify flash-attn is accessible
if python -c "import flash_attn" 2>/dev/null; then
    echo "✓ flash-attn found and importable"
else
    echo "ERROR: flash-attn not found in venv"
    exit 1
fi

echo "=== Venv setup complete ==="
