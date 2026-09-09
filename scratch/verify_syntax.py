
with open('weather-app/js/app.js', 'r', encoding='utf-8') as f:
    code = f.read()

import tokenize, io
print('Total lines in app.js:', len(code.splitlines()))
print('Ends with: ', repr(code[-40:]))
assert code.strip().endswith('})();'), 'File should end with })();'
print('IIFE closure verified!')
