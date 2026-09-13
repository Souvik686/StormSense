from playwright.sync_api import sync_playwright
import time

def test():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        errors = []
        page.on('pageerror', lambda e: errors.append(f'PAGE ERROR: {e}'))
        
        try:
            page.goto('http://127.0.0.1:8000/', timeout=10000)
            page.wait_for_selector('#pipeline-status-text', timeout=10000)
            print('Testing Live mode horizon clicks...')
            for btn in [2, 4, 6]:
                page.click(f'button[onclick=\"window.setForecastHorizon(this, \'+{btn}h\', {btn})\"]')
                time.sleep(1)
                
            print('Toggling to Historical mode...')
            page.click('#btn-mode-toggle')
            time.sleep(3)
            
            print('Testing Historical mode horizon clicks...')
            for btn in [2, 4, 6]:
                page.click(f'button[onclick=\"window.setForecastHorizon(this, \'+{btn}h\', {btn})\"]')
                time.sleep(1)
                
            print('--- ERRORS ---')
            for e in errors:
                print(e)
            print('SUCCESS')
                
        except Exception as e:
            print(f'Exception: {e}')
        finally:
            browser.close()

test()
