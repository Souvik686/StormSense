import os,sys,json
os.environ["STORMSENSE_DEVICE"]="cpu"
sys.path.insert(0,os.path.abspath("."))
from src.inference.predictor import get_predictor
p=get_predictor("Data/outputs/checkpoints/v2_calibrated_best.pt",None,device="cpu")
info=p.model_info()
info["calibrated_temperatures"]=p.temperature
info["calibrated_thresholds"]=p.threshold_per_lead
print("raw dict built OK:")
for k,v in info.items(): print(f"   {k}: {type(v).__name__} = {str(v)[:70]}")
print("\nJSON-serializable?")
try:
    json.dumps(info); print("   stdlib json: OK")
except Exception as e: print("   stdlib json FAILED:",e)
# FastAPI uses its own encoder
from fastapi.encoders import jsonable_encoder
try:
    jsonable_encoder(info); print("   jsonable_encoder: OK")
except Exception as e: print("   jsonable_encoder FAILED:",type(e).__name__,e)
import math
print("\nbest_val_loss value:",repr(info["best_val_loss"]),
      "isnan:", (isinstance(info["best_val_loss"],float) and math.isnan(info["best_val_loss"])))
