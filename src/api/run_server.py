"""Server entry point for the FastAPI backend.

Usage:
    python -m src.api.run_server
    -- or --
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Environment variables (all optional):
    CHECKPOINT_PATH   path to trained .pt checkpoint (default: outputs/checkpoints/best.pt)
    CONFIG_PATH       path to YAML config (default: configs/default.yaml)
    API_KEY           if set, all /predict calls require ?x_api_key=<value>
    HOST              server bind host (default: 0.0.0.0)
    PORT              server port (default: 8000)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def main():
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not found. Install with: pip install uvicorn")
        sys.exit(1)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "0") == "1"

    print(f"Starting SevereWeatherNet API on {host}:{port}")
    print(f"  Checkpoint : {os.getenv('CHECKPOINT_PATH', 'outputs/checkpoints/best.pt')}")
    print(f"  Config     : {os.getenv('CONFIG_PATH', 'configs/default.yaml (default)')}")
    print(f"  API Key    : {'set' if os.getenv('API_KEY') else 'not set (open access)'}")
    print(f"  Docs       : http://{host}:{port}/docs")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

