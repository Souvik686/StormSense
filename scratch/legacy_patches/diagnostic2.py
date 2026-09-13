import numpy as np

from src.inference.nowcast_service import get_nowcast_service

def run():
    svc = get_nowcast_service()
    svc.refresh_live_state()
    pred = svc.live_pred
    
    h = 2
    li = svc.lead_times.index(h)
    
    prob_grid = pred["severe_weather_prob"][li]
    rain_grid = pred["rain_3h_mm_pred"][li]
    
    p_idx = np.unravel_index(np.argmax(prob_grid), prob_grid.shape)
    r_idx = np.unravel_index(np.argmax(rain_grid), rain_grid.shape)
    
    print(f"SEVERE PROB MAX: {svc.lats[p_idx[0]]:.3f}°N, {svc.lons[p_idx[1]]:.3f}°E ({prob_grid[p_idx]:.3f})")
    print(f"RAIN PREDICT MAX: {svc.lats[r_idx[0]]:.3f}°N, {svc.lons[r_idx[1]]:.3f}°E ({rain_grid[r_idx]:.3f} mm)")

run()
