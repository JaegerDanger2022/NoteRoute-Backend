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

print("Starting uvicorn...", flush=True)
import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=2)
