# update_js.py
import re

with open('weather-app/js/app.js.bak', 'r', encoding='utf-8') as f:
    orig = f.read()

print('Read backup successfully, length:', len(orig))
