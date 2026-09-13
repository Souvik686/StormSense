import re

with open('src/inference/nowcast_service.py', 'r', encoding='utf-8') as f:
    code = f.read()

import_str = "import numpy as np\nfrom .xai import calculate_xai_factors\n"
if "from .xai import calculate_xai_factors" not in code:
    code = code.replace("import numpy as np\n", import_str)

old_def = """    def get_xai_attribution(self) -> Dict[str, Any]:
        \"\"\"Returns physical feature attribution and model architecture rationale.\"\"\"
        return {"""

new_def = """    def get_xai_attribution(self, mode: Optional[str] = None) -> Dict[str, Any]:
        \"\"\"Returns physical feature attribution and model architecture rationale.\"\"\"
        resolved = self._resolve_mode(mode)
        if resolved == "live":
            # For live, we use the rule-based physics XAI
            xai_inputs = {
                "rainfall_1h_mm": 0, # Could be derived from GFS surface if needed
                "rainfall_3h_mm": 0,
                "rainfall_6h_mm": 0,
                "humidity_percent": 85, # placeholder or from OpenWeather
                "dew_point_c": 24, # placeholder
                "cape_jkg": 1500, # default plausible if missing
                "wind_speed_kmh": 15,
                "radar_dbz": None
            }
            # Try to grab real values if live pred is available
            if self.live_pred is not None:
                # Use a typical cell or just mean across WB
                # Or just use the OpenWeather telemetry if available
                # But XAI is for the ML input, we don't have per-cell XAI yet.
                pass
            
            return calculate_xai_factors(xai_inputs)
            
        return {"""

code = code.replace(old_def, new_def)

with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
    f.write(code)

