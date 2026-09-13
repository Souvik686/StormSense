import os

with open('src/inference/nowcast_service.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Inject threading lock
import_str = 'import threading\\n'
if import_str not in code:
    code = code.replace('import os\\n', 'import os\\n' + import_str)

old_def = '_SERVICE_INSTANCE: Optional[NowcastService] = None'
new_def = '_SERVICE_LOCK = threading.Lock()\\n_SERVICE_INSTANCE: Optional[NowcastService] = None'
code = code.replace(old_def, new_def)

old_func = '''def get_nowcast_service() -> NowcastService:
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = NowcastService()
    return _SERVICE_INSTANCE'''

new_func = '''def get_nowcast_service() -> NowcastService:
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        with _SERVICE_LOCK:
            if _SERVICE_INSTANCE is None:
                _SERVICE_INSTANCE = NowcastService()
    return _SERVICE_INSTANCE'''
    
code = code.replace(old_func, new_func)

with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Fixed NowcastService lifecycle concurrency bug.')
