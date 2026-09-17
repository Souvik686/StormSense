import re
import numpy as np

with open('src/inference/nowcast_service.py', encoding='utf-8') as f:
    text = f.read()

replacement = """
                # We need T=0 risk as well. The model outputs for +2h...+6h.
                # To get risk valid at T=0, we run inference on the sample from T-2h.
                sample_t2 = dataset[sample_idx - 2]
                
                def _infer(s):
                    with torch.no_grad():
                        batch = {
                            "surface": s["surface"][None].to(self.predictor.device),
                            "pressure_wind": s["pressure_wind"][None].to(self.predictor.device),
                            "pressure_thermo": s["pressure_thermo"][None].to(self.predictor.device),
                            "dem": s["dem"][None].to(self.predictor.device),
                        }
                        return self.predictor.model(batch)

                preds = _infer(sample)
                preds_now = _infer(sample_t2)

                def _get_prob(p):
                    logits = p["severe_weather_logit"][0]
                    if self.predictor.temperature is not None:
                        T = torch.tensor(self.predictor.temperature, device=logits.device, dtype=logits.dtype).view(-1, 1, 1)
                        return torch.sigmoid(logits / T).cpu().numpy()
                    return torch.sigmoid(logits).cpu().numpy()

                severe_prob = _get_prob(preds)
                severe_prob_now = _get_prob(preds_now)
                
                # Prepend the +2h output from T-2h prediction as the T=0 output for the current prediction
                severe_prob = np.concatenate([severe_prob_now[0:1], severe_prob], axis=0)

                rain_pred = preds["rain_3h_mm"][0].cpu().numpy()
                rain_pred_now = preds_now["rain_3h_mm"][0].cpu().numpy()
                rain_pred = np.concatenate([rain_pred_now[0:1], rain_pred], axis=0)

                # Binary classification using calibrated per-lead thresholds
                if self.predictor.threshold_per_lead is not None:
                    severe_binary = np.zeros_like(severe_prob, dtype=np.uint8)
                    for li, lh in enumerate(self.lead_times):
                        thr = float(self.predictor.threshold_per_lead.get(lh, self.predictor.threshold))
                        severe_binary[li] = (severe_prob[li] >= thr).astype(np.uint8)
                else:
                    severe_binary = (severe_prob >= self.predictor.threshold).astype(np.uint8)

                # Flash-flood/overall compound risk (shared formula, see derive_compound_risks)
                dem_norm = sample["dem"][0].numpy()  # (33, 25)
                self._live_dem_norm = dem_norm  # DEM is static; reused for live inference too
                compound = derive_compound_risks(severe_prob, rain_pred, dem_norm)

                self.current_pred = {
"""

# Replace the block
start = text.find("with torch.no_grad():")
end = text.find("self.current_pred = {")
if start != -1 and end != -1:
    text = text[:start] + replacement.strip() + "\n                self.current_pred = {" + text[end+21:]
    with open('src/inference/nowcast_service.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Patched _init_operational_state")
else:
    print("Could not find block in _init_operational_state")

