import re

with open('frontend/js/app.js', encoding='utf-8') as f:
    text = f.read()

text = text.replace('if (lead === "now" || lead === "NOW" || lead == null) return 2;', 'if (lead === "now" || lead === "NOW" || lead == null) return 0;')
text = text.replace('return (n === 2 || n === 3 || n === 4 || n === 5 || n === 6) ? n : 2;', 'return (n === 0 || n === 2 || n === 3 || n === 4 || n === 5 || n === 6) ? n : 0;')

with open('frontend/js/app.js', 'w', encoding='utf-8') as f:
    f.write(text)

