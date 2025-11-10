cat > ~/repos/open-r1_safety/openr1/bin/python-glibc << 'EOF'
#!/bin/bash
GLIBC_NEW=/home/user/glibc
PYTHON_DIR="$(dirname "$0")"
PYTHON_LIB="$(dirname "$PYTHON_DIR")/lib"
GCC_LIBDIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"

exec /home/user/glibc/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/user/glibc/lib:$PYTHON_LIB:$GCC_LIBDIR:${CUDA_HOME:+$CUDA_HOME/lib64}" \
  "$PYTHON_DIR/python" "$@"
EOF

chmod +x ~/repos/open-r1_safety/openr1/bin/python-glibc