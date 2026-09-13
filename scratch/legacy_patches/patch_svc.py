import re
with open('src/inference/nowcast_service.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Modify refresh_live_state signature
old_sig = '''    def refresh_live_state(self) -> bool:'''
new_sig = '''    def refresh_live_state(self, target_t0=None) -> bool:'''
code = code.replace(old_sig, new_sig)

# Modify harmonized call
old_harm = '''            harmonized = gfs_live.fetch_and_harmonize(self.lats, self.lons)'''
new_harm = '''            from datetime import datetime, timezone
            if target_t0 is None:
                target_t0 = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            harmonized = gfs_live.fetch_and_harmonize(self.lats, self.lons, target_t0=target_t0)'''
code = code.replace(old_harm, new_harm)

with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched nowcast_service.py")
