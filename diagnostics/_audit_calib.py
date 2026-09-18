import numpy as np
T=[0.4152,0.4144,0.4169,0.4105,0.3819]
def sig(x): return 1/(1+np.exp(-x))
print("Effect of temperature scaling (T<1 sharpens):")
print(f"{'raw_p':>8} {'logit':>8} " + " ".join(f"T={t}" for t in T))
for raw in [0.05,0.10,0.20,0.25,0.30,0.40,0.50,0.60,0.75,0.90]:
    lg=np.log(raw/(1-raw))
    cal=[sig(lg/t) for t in T]
    print(f"{raw:8.2f} {lg:8.3f} " + " ".join(f"{c:7.4f}" for c in cal))
print()
print("Inverse: what RAW prob is needed to display >=25% / 50% / 75% after calibration (T=0.4152, lead2):")
for disp in [0.25,0.50,0.75]:
    lg_cal=np.log(disp/(1-disp))
    raw=sig(lg_cal*0.4152)
    print(f"  display {disp:.2f} <- raw {raw:.4f}")
print()
print("Model's OWN operating thresholds vs UI display bands:")
thr={2:0.727,3:0.679,4:0.673,5:0.637,6:0.625}
for k,v in thr.items():
    print(f"  lead {k}h: decision threshold {v} -> UI band = " + ("ALERT/orange" if v<0.75 else "WARNING"))
