import os

filepath = 'src/inference/nowcast_service.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('format_iso_time(self.live_anchor_t0) if self.live_anchor_t0 else None', 'format_iso_time(self.live_valid_time) if self.live_valid_time else None')
content = content.replace('"input_slot_provenance": self.live_provenance,', '"input_slot_provenance": self.live_slot_provenance,')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed nowcast_service.py variables")
