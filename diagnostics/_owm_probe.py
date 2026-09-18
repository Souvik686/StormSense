import os,sys,json,asyncio,time
sys.path.insert(0,os.path.abspath("."))
from datetime import datetime,timezone,timedelta
from dotenv import load_dotenv; load_dotenv(".env")
import httpx
KEY=os.getenv("OPENWEATHER_API_KEY")
LOCS={"Kolkata":(22.5726,88.3639),"Darjeeling":(27.0410,88.2663),"Purulia":(23.3322,86.3616),
      "Digha":(21.6270,87.5090),"Malda":(25.0000,88.1500),"Siliguri":(26.7271,88.3953)}
async def main():
    async with httpx.AsyncClient(timeout=20) as c:
        for nm,(la,lo) in LOCS.items():
            try:
                r=await c.get("https://api.openweathermap.org/data/2.5/weather",
                    params={"lat":la,"lon":lo,"appid":KEY,"units":"metric"})
                d=r.json()
                if r.status_code!=200: print(nm,"HTTP",r.status_code,d); continue
                dt=datetime.fromtimestamp(d["dt"],timezone.utc)
                age=(datetime.now(timezone.utc)-dt).total_seconds()/60
                st=d.get("name"); rain=d.get("rain",{}).get("1h",0.0)
                print(f"{nm:11s} station='{st}' obs={dt.strftime('%H:%M UTC')} age={age:5.1f}min "
                      f"T={d['main']['temp']:5.1f}C RH={d['main']['humidity']:3d}% "
                      f"p={d['main']['pressure']}hPa rain1h={rain} wx={d['weather'][0]['main']}")
                if nm=="Kolkata": print("   RAW KEYS:",sorted(d.keys()))
            except Exception as e: print(nm,"ERR",e)
asyncio.run(main())
