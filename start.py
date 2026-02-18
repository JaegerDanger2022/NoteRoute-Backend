import sys
import traceback

print("=== NoteRoute Backend Starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)

print("Testing app import...", flush=True)
try:
    import app.main
    print("Import OK", flush=True)
except Exception as e:
    print(f"IMPORT FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

import os
port = int(os.environ.get("PORT", 8000))
print(f"Starting uvicorn on port {port}...", flush=True)
import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=port, workers=2)
