#!/bin/bash
# fix_venv.sh - Repair uv/standard venv symlinks for current pod

set -e

VENV_PATH="${1:-.venv}"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "Error: '$VENV_PATH' is not a directory"
    exit 1
fi

if [[ ! -f "$VENV_PATH/pyvenv.cfg" ]]; then
    echo "Error: '$VENV_PATH' does not appear to be a venv (no pyvenv.cfg)"
    exit 1
fi

# Extract Python version - handle both standard (version) and uv (version_info) formats
VENV_VERSION=$(grep -E "^version(_info)?\s*=" "$VENV_PATH/pyvenv.cfg" | head -1 | sed 's/.*=\s*//' | cut -d. -f1,2)

if [[ -z "$VENV_VERSION" ]]; then
    echo "Warning: Could not detect version from pyvenv.cfg, trying to infer from lib/"
    VENV_VERSION=$(ls "$VENV_PATH/lib/" | grep -oP 'python\K[0-9]+\.[0-9]+' | head -1)
fi

echo "Detected venv Python version: $VENV_VERSION"

if [[ -z "$VENV_VERSION" ]]; then
    echo "Error: Could not determine Python version"
    exit 1
fi

# Find a matching Python on this system
PYTHON_BIN=""
for candidate in "python$VENV_VERSION" "python3" "python"; do
    if command -v "$candidate" &>/dev/null; then
        CANDIDATE_VERSION=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$CANDIDATE_VERSION" == "$VENV_VERSION" ]]; then
            PYTHON_BIN=$(command -v "$candidate")
            break
        fi
    fi
done

# Also check uv-managed pythons
if [[ -z "$PYTHON_BIN" ]]; then
    UV_PYTHON="$HOME/.local/share/uv/python/cpython-${VENV_VERSION}."*"-linux-x86_64-gnu/bin/python3"
    for p in $UV_PYTHON; do
        if [[ -x "$p" ]]; then
            PYTHON_BIN="$p"
            break
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: Could not find Python $VENV_VERSION on this system"
    echo "Try: uv python install $VENV_VERSION"
    exit 1
fi

PYTHON_HOME=$(dirname "$PYTHON_BIN")
PYTHON_REAL=$(readlink -f "$PYTHON_BIN")

echo "Using Python: $PYTHON_BIN (resolves to $PYTHON_REAL)"
echo "Python home: $PYTHON_HOME"

# Update pyvenv.cfg (atomic write via temp file + mv)
CFG="$VENV_PATH/pyvenv.cfg"
TMP_CFG=$(mktemp)
sed "s|^home\s*=.*|home = $PYTHON_HOME|" "$CFG" > "$TMP_CFG"
mv "$TMP_CFG" "$CFG"
echo "Updated pyvenv.cfg"

# Fix symlinks in bin/ - ln -sf is atomic, won't break running processes
BIN_DIR="$VENV_PATH/bin"

# Remove broken python symlinks and recreate
for link in python python3 "python$VENV_VERSION"; do
    target="$BIN_DIR/$link"
    if [[ -L "$target" ]] || [[ ! -e "$target" ]]; then
        ln -sf "$PYTHON_REAL" "$target"
        echo "Fixed symlink: $link -> $PYTHON_REAL"
    fi
done

echo ""
echo "Done! Test with:"
echo "  source $VENV_PATH/bin/activate"
echo "  which python"
echo "  python --version"
