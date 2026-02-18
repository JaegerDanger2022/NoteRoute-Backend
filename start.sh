#!/bin/sh
set -e

echo "=== NoteRoute Backend Starting ==="
echo "Python: $(python --version)"
echo "Testing app import..."

python -c "
import sys
try:
    import app.main
    print('Import OK')
except Exception as e:
    print(f'IMPORT FAILED: {type(e).__name__}: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
