import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.inference.predictor import NowcastPredictor

def run():
    p = NowcastPredictor('Data/outputs/checkpoints/v2_calibrated_best.pt', 'configs/default.yaml', device='cpu')
    
    surface = np.full((6, 9, 33, 25), np.nan, dtype=np.float32)
    pressure = np.full((6, 5, 6, 33, 25), np.nan, dtype=np.float32)
    
    # Valid data
    surface[:, :] = 1.0
    pressure[:, :] = 1.0
    
    # Missing levels (like ERA5 missing data)
    pressure[:, 2:5, 0, :, :] = np.nan
    
    dem = np.zeros((33, 25), dtype=np.float32)
    
    out = p.predict(surface, pressure, dem, "2026-09-11T12:00:00Z")
    
    print("SURFACE SHAPE:", surface.shape)
    print("PRESSURE SHAPE:", pressure.shape)
    print("PRESSURE NANS IN:", np.isnan(pressure).sum())
    print("SEVERE PROB SHAPE:", out["severe_weather_prob"].shape)
    print("SEVERE PROB NANS OUT:", np.isnan(out["severe_weather_prob"]).sum())
    print("SEVERE PROB MAX:", float(np.max(out["severe_weather_prob"])))
    print("NAN_SAFETY: PASS" if np.isnan(out["severe_weather_prob"]).sum() == 0 else "NAN_SAFETY: FAIL")

run()
