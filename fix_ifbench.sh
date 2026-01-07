# Find the file
IFBENCH_FILE=$(python -c "import lighteval.tasks.tasks.ifbench.instructions as m; print(m.__file__)")

# Patch it (add check for empty words list)
sed -i 's/if words\[0\] != words\[-1\]:/if words and words[0] != words[-1]:/' "$IFBENCH_FILE"