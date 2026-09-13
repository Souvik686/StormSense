import sys
import os
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.abspath("."))
from src.inference.gfs_live import fetch_and_harmonize

# Mock lats and lons
lats = np.linspace(28.0, 20.0, 33)
lons = np.linspace(84.0, 90.0, 25)

now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
res = fetch_and_harmonize(lats, lons, target_t0=now)
print("Harmonized t0:", res.t0)
print("Slot timestamps:", res.slot_timestamps)
print("Surface shape:", res.surface.shape)
