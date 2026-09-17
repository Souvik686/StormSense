import re

with open('src/inference/nowcast_service.py', encoding='utf-8') as f:
    text = f.read()

replacement = """
            import datetime
            from datetime import timedelta
            
            harmonized = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0)
            preds_dict = self.predictor.predict(
                surface=harmonized.surface,
                pressure=harmonized.pressure,
                dem=self._get_static_dem_meters(),
                timestamp=harmonized.t0.isoformat(),
            )
            
            # Fetch for T-2h to get the valid T=0 prediction (lead 2)
            harmonized_now = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0 - timedelta(hours=2))
            preds_dict_now = self.predictor.predict(
                surface=harmonized_now.surface,
                pressure=harmonized_now.pressure,
                dem=self._get_static_dem_meters(),
                timestamp=harmonized_now.t0.isoformat(),
            )
            
            import numpy as np
            severe_prob = np.concatenate([preds_dict_now["severe_weather_prob"][0:1], preds_dict["severe_weather_prob"]], axis=0)
            rain_pred = np.concatenate([preds_dict_now["rain_3h_mm_pred"][0:1], preds_dict["rain_3h_mm_pred"]], axis=0)
            severe_binary = np.concatenate([preds_dict_now["severe_weather_binary"][0:1], preds_dict["severe_weather_binary"]], axis=0)
            
            dem_norm = self._live_dem_norm if self._live_dem_norm is not None else np.zeros(
                (len(self.lats), len(self.lons)), dtype=np.float32
            )
            compound = derive_compound_risks(severe_prob, rain_pred, dem_norm)

            self.live_pred = {
"""

start = text.find("harmonized = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0)")
end = text.find("self.live_pred = {", start)
if start != -1 and end != -1:
    text = text[:start] + replacement.strip() + "\n            self.live_pred = {" + text[end+18:]
    with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Patched refresh_live_state")
else:
    print("Could not find block in refresh_live_state")

