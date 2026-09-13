import asyncio
import httpx
import time

async def fetch_live(client, i):
    t0 = time.time()
    try:
        r = await client.post("http://127.0.0.1:8000/api/live/refresh", timeout=60.0)
        return i, r.status_code, time.time() - t0
    except Exception as e:
        return i, str(e), time.time() - t0

async def main():
    print("Starting concurrency test with 5 simultaneous requests...")
    async with httpx.AsyncClient() as client:
        tasks = [fetch_live(client, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        for i, status, dt in results:
            print(f"Req {i}: status={status} time={dt:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
