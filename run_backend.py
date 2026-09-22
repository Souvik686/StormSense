"""StormSense backend-only API server (SPLIT MODE).

Runs the FastAPI application without expecting to be the origin that serves the
dashboard. It is the SAME app object as merged mode -- `backend/main.py` -- so
there is exactly one implementation of the ML, GFS and historical services and
no logic is duplicated per mode.

    Terminal 1:  python run_backend.py      # API on :8000
    Terminal 2:  python run_frontend.py     # dashboard on :3000

Merged single-command mode remains available and unchanged:

    python run_server.py                    # dashboard + API on :8000

The static-file mounts in backend/main.py stay active here, so this server can
still serve the dashboard directly; that is harmless and keeps merged mode and
split mode behaviourally identical at the API level.

Environment:
    HOST                     default 127.0.0.1
    PORT                     default 8000
    STORMSENSE_CORS_ORIGINS  comma-separated allowlist (see backend/main.py)
    STORMSENSE_DEVICE        set to "cpu" on memory-constrained machines
"""
import importlib.util
import os
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

backend_path = os.path.join(repo_root, "backend", "main.py")
spec = importlib.util.spec_from_file_location("unified_backend", backend_path)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
app = _mod.app

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    origins = getattr(_mod, "CORS_ALLOW_ORIGINS", [])
    print("=" * 68)
    print("  StormSense BACKEND API (split mode)")
    print(f"  API            : http://{host}:{port}/api")
    print(f"  Health         : http://{host}:{port}/api/health")
    print(f"  Docs           : http://{host}:{port}/docs")
    print(f"  CORS allowlist : {', '.join(origins) if origins else '(none)'}")
    print("  Start the dashboard separately:  python run_frontend.py")
    print("=" * 68)
    uvicorn.run(app, host=host, port=port, log_level="info")
