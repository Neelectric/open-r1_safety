# fix-venv.sh
#!/bin/bash
cd /workspace/writeable/repos/open-r1_safety/openr1/bin
rm -f python python3 python3.*
ln -s $(which python3) python3
ln -s python3 python