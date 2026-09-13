from playwright.sync_api import sync_playwright
import time

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('http://127.0.0.1:8000')
        time.sleep(2)
        
        print("Checking Live Mode...")
        # Check NOW
        page.locator('button[onclick="window.setForecastHorizon(this, \'now\', 0)"]').click()
        time.sleep(1)
        print("LIVE NOW: PASS")
        
        for h in [2, 4, 6]:
            page.locator(f'button[onclick="window.setForecastHorizon(this, \'+{h}h\', {h})"]').click()
            time.sleep(1)
            print(f"LIVE +{h}h: PASS")
            
        print("Switching to Historical Mode...")
        page.click('#btn-mode-toggle')
        time.sleep(2)
        
        print("Checking Historical Mode...")
        page.locator('button[onclick="window.setForecastHorizon(this, \'now\', 0)"]').click()
        time.sleep(1)
        print("HISTORICAL NOW: PASS")
        for h in [2, 4, 6]:
            page.locator(f'button[onclick="window.setForecastHorizon(this, \'+{h}h\', {h})"]').click()
            time.sleep(1)
            print(f"HISTORICAL +{h}h: PASS")
            
        print("BROWSER: PASS")
        browser.close()
run()
