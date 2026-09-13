"""
StormSense Mission Control - Unified Server Runner
Direct runner serving both the StormSense frontend and the SevereWeatherNet V2 ML backend.

Usage:
    python run_server.py
    - or -
    python backend/main.py
"""
import importlib.util
import os
import sys

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

backend_path = os.path.join(repo_root, 'backend', 'main.py')
spec = importlib.util.spec_from_file_location('unified_backend', backend_path)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
app = _mod.app

if __name__ == '__main__':
    import uvicorn
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '8000'))
    print('=' * 68)
    print('  StormSense Mission Control & SevereWeatherNet V2 Nowcasting')
    print(f'  Serving Dashboard + API at: http://{host}:{port}')
    print(f'  API Health Check:           http://{host}:{port}/api/health')
    print(f'  Interactive API Docs:       http://{host}:{port}/docs')
    print('=' * 68)
    uvicorn.run(app, host=host, port=port, log_level='info')
