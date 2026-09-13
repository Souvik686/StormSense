import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PAGE ERROR: {err}"))
        
        print("Navigating to local server...")
        await page.goto("http://127.0.0.1:8000")
        
        await page.wait_for_timeout(3000)
        print("Page loaded.")
        
        await browser.close()
        print("Done.")

if __name__ == '__main__':
    asyncio.run(run())
