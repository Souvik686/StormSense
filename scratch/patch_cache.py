import re

with open('src/inference/gfs_live.py', encoding='utf-8') as f:
    text = f.read()

replacement = """                        cycle.pressure[var_key][lvl] = arr

                try:
                    import pickle
                    with open(cache_file, 'wb') as f_cache:
                        pickle.dump(cycle, f_cache)
                except Exception:
                    pass
                return cycle"""

text = text.replace('                        cycle.pressure[var_key][lvl] = arr\n\n                return cycle', replacement)

with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(text)

