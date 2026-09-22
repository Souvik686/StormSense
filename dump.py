import traceback, threading, time, sys, os
sys.path.insert(0, os.path.abspath('.'))
def dump_trace():
    time.sleep(10)
    print('--- TRACE ---')
    for th_id, frame in sys._current_frames().items():
        print(f'Thread {th_id}:')
        traceback.print_stack(frame)
    os._exit(1)
t = threading.Thread(target=dump_trace)
t.daemon = True
t.start()
from src.inference.nowcast_service import get_nowcast_service
print('Instantiating...')
svc = get_nowcast_service()
print('Done!')

