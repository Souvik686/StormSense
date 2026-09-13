import sys
import os
import numpy as np
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.abspath("."))
from src.inference.gfs_live import find_latest_gfs_cycle

cycle = find_latest_gfs_cycle()
print("Latest GFS Cycle:", cycle)
