import os

with open('src/inference/nowcast_service.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_print = 'print(f"[NowcastService] Operational state loaded. Valid time: {self.current_valid_time}")'
new_print = 'print(f"[NowcastService] Historical (2024-05-05) operational state loaded. Valid time: {self.current_valid_time}")'
code = code.replace(old_print, new_print)

old_print2 = 'print(f"[NowcastService] Precomputed continuous West Bengal risk surfaces'
new_print2 = 'print(f"[NowcastService] Precomputed historical continuous West Bengal risk surfaces'
code = code.replace(old_print2, new_print2)

old_issue = '''    def _issue_time_for_mode(self, mode: Optional[str]) -> str:
        m = self._resolve_mode(mode)
        if m == "live":
            return self.live_valid_time or self.current_valid_time
        return self.current_valid_time'''

new_issue = '''    def _issue_time_for_mode(self, mode: Optional[str]) -> str:
        m = self._resolve_mode(mode)
        if m == "live":
            return self.live_valid_time or "Unknown"
        return self.current_valid_time'''
code = code.replace(old_issue, new_issue)

with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Fixed timestamp leakage and logging.')
