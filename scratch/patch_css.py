with open('frontend/css/styles.css', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('.gm-style img {\n    image-rendering: pixelated;\n    image-rendering: crisp-edges;\n}', '')
text += '\nimg[src*="api/nowcast/risk-surface"], img[src*="api/observations/surface"] {\n    image-rendering: pixelated;\n    image-rendering: crisp-edges;\n}\n'
with open('frontend/css/styles.css', 'w', encoding='utf-8') as f:
    f.write(text)

