from playwright.sync_api import sync_playwright
import time

def test():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        errors = []
        page.on('pageerror', lambda e: errors.append(f'PAGE ERROR: {e}'))
        page.on('console', lambda msg: errors.append(f'CONSOLE {msg.type}: {msg.text}') if msg.type in ['error', 'warning'] else None)
        
        try:
            page.goto('http://127.0.0.1:8000/', timeout=10000)
            page.wait_for_selector('#pipeline-status-text', timeout=5000)
            time.sleep(2)
            
            pipe_text = page.locator('#pipeline-status-text').inner_text()
            print(f'Pipeline Status Text: {pipe_text}')
            
            clock_text = page.locator('#header-clock-ist').inner_text()
            print(f'Clock Text: {clock_text}')
            
            obs_text = page.locator('#header-obs-time-text').inner_text()
            print(f'Observed Text: {obs_text}')
            
            print('--- ERRORS ---')
            for e in errors:
                print(e)
                
        except Exception as e:
            print(f'Exception: {e}')
        finally:
            browser.close()

test()
