import sys
import os

filepath = 'src/inference/nowcast_service.py'
with open(filepath, 'r', encoding='utf-8') as f:
    code = f.read()

# Replace hardcoded valid time init
code = code.replace(
    'self.current_valid_time: str = "2024-05-05T15:00:00Z"',
    'self.current_valid_time: str = "2024-05-26T12:00:00Z"'
)

# Find where it selects the sample
old_sample_code = '''                loaders = make_dataloaders_v2(cache_path, self.cfg)
                # Sample 100: Active Kalbaishakhi pre-monsoon convective squall in 2024 test split
                dataset = loaders["test"].dataset
                sample_idx = min(100, len(dataset) - 1)
                sample = dataset[sample_idx]'''

new_sample_code = '''                loaders = make_dataloaders_v2(cache_path, self.cfg)
                import numpy as np
                # Historical Case Study: Cyclone Remal (May 26, 2024, 12:00 UTC)
                dataset = loaders["test"].dataset
                target_time = np.datetime64("2024-05-26T12:00:00")
                sample_idx = 100 # fallback
                for i, w in enumerate(dataset.windows):
                    vt = dataset.times[w.input_end_idx]
                    if vt == target_time:
                        sample_idx = i
                        break
                sample = dataset[sample_idx]'''

if old_sample_code in code:
    code = code.replace(old_sample_code, new_sample_code)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched nowcast_service.py successfully.")
else:
    print("Could not find old_sample_code in nowcast_service.py")
