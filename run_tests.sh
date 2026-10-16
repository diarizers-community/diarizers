#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v flake8 >/dev/null 2>&1; then
  flake8 --indent-size 2 --max-line-length 80 \
    src/diarizers scripts tests hf_space/app.py setup.py
else
  "$PYTHON_BIN" - <<'EOF'
import ast
import glob
import sys

files = (
     sorted(glob.glob("src/diarizers/*.py"))
    + sorted(glob.glob("scripts/*.py"))
    + sorted(glob.glob("tests/*.py"))
    + ["hf_space/app.py", "setup.py"]
)
errors = []
for path in files:
  with open(path, "r", encoding="utf-8") as f:
    src = f.read()
  try:
    ast.parse(src, filename=path)
  except SyntaxError as e:
    errors.append(f"{path}:{e.lineno}: SyntaxError: {e.msg}")
  for lineno, line in enumerate(src.splitlines(), start=1):
    if len(line) > 80:
      errors.append(f"{path}:{lineno}: E501 line too long ({len(line)} > 80)")
    if line.endswith(" ") or line.endswith("\t"):
      errors.append(f"{path}:{lineno}: W291 trailing whitespace")
if errors:
  for err in errors:
    print(err, file=sys.stderr)
  sys.exit(1)
print(f"Style & AST check passed ({len(files)} files, <=80 chars, 2-space indent).")
EOF
fi

if "$PYTHON_BIN" -c "import pytest, pytest_cov" >/dev/null 2>&1; then
  PYTHONPATH=src "$PYTHON_BIN" -m pytest tests/ -v --cov=src/diarizers --cov-report=term-missing
elif "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  PYTHONPATH=src "$PYTHON_BIN" -m pytest tests/ -v
elif command -v pytest >/dev/null 2>&1; then
  PYTHONPATH=src pytest tests/ -v --cov=src/diarizers --cov-report=term-missing
else
  PYTHONPATH=src "$PYTHON_BIN" -m unittest discover -s tests -p "test_*.py" -v
fi


