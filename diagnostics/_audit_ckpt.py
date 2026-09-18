import torch, os
p="Data/outputs/checkpoints/v2_calibrated_best.pt"
print("exists:", os.path.exists(p))
ck=torch.load(p,map_location="cpu",weights_only=False)
print("keys:", list(ck.keys()))
print("optimal_threshold:", ck.get("optimal_threshold"))
print("threshold_per_lead:", ck.get("threshold_per_lead"))
print("temperature_per_lead:", ck.get("temperature_per_lead"))
print("is_calibrated:", ck.get("is_calibrated"))
print("epoch:", ck.get("epoch"), "best_val_loss:", ck.get("best_val_loss"))
print("param count:", sum(v.numel() for v in ck["model"].values()))
