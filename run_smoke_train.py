import torch
import os
import sys
from src.utils.config import load_config
from src.data.dataset import make_dataloaders
from src.models.advanced import SevereWeatherNet
from src.training.losses import MultiTaskLoss
from src.features.normalize import SINGLE_VARS, PRESSURE_VARS

cfg = load_config()
cache_path = os.path.join(cfg.path("paths", "cache_root"), "era5_memmap")
loaders = make_dataloaders(cache_path, cfg)
train_loader = loaders["train"]

print(f"Dataset sizes: train={len(loaders['train'].dataset)}, val={len(loaders['val'].dataset)}")
print(f"Number of batches: train={len(train_loader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SevereWeatherNet(
    n_surface_vars=len(SINGLE_VARS),
    n_pressure_vars=len(PRESSURE_VARS),
    n_levels=6,
    n_lead_times=5,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = MultiTaskLoss()

print("\nStarting end-to-end smoke training for 2 batches...")
model.train()
for i, batch in enumerate(train_loader):
    if i >= 2:
        break
    
    # Move to device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    
    optimizer.zero_grad()
    preds = model(batch)
    losses = loss_fn(preds, batch)
    loss = losses["total"]
    
    loss.backward()
    optimizer.step()
    
    print(f"Batch {i+1} | Loss: {loss.item():.4f}")

print("\nSmoke training completed successfully!")

