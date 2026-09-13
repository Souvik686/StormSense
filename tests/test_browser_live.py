import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to local server...")
        await page.goto("http://127.0.0.1:8000")
        
        await page.wait_for_timeout(3000)
        print("Page loaded.")
        
        # Click NOW
        await page.locator(".horizon-btn:has-text('NOW')").click()
        await page.wait_for_timeout(1000)
        
        ts = await page.locator("#nv-obs-time").text_content()
        print(f"NOW displayed timestamp: {ts}")
        
        # Click +2h
        await page.locator(".horizon-btn:has-text('+2h')").click()
        await page.wait_for_timeout(1500)
        
        vt_2h = await page.locator("#fv-valid-time").text_content()
        print(f"+2h valid timestamp: {vt_2h}")
        
        # Click +6h
        await page.locator(".horizon-btn:has-text('+6h')").click()
        await page.wait_for_timeout(1500)
        
        vt_6h = await page.locator("#fv-valid-time").text_content()
        print(f"+6h valid timestamp: {vt_6h}")
        
        
        
        await browser.close()
        print("Browser automation tests completed.")

if __name__ == '__main__':
    asyncio.run(run())
