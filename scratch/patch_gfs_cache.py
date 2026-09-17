import os

with open('src/inference/gfs_live.py', encoding='utf-8') as f:
    text = f.read()

cache_code = """
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "processed", "cache", "gfs")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cycle_time.strftime('%Y%m%d_%H%M%S')}_{f_hour}.pkl")
    if os.path.exists(cache_file):
        try:
            import pickle
            with open(cache_file, 'rb') as f_cache:
                return pickle.load(f_cache)
        except Exception as e:
            pass
"""

save_code = """
                try:
                    import pickle
                    with open(cache_file, 'wb') as f_cache:
                        pickle.dump(cycle, f_cache)
                except Exception:
                    pass
                return cycle
"""

text = text.replace(
    "    last_err: Optional[Exception] = None\n    for base in (AWS_BASE, NOMADS_BASE):", 
    cache_code + "    last_err: Optional[Exception] = None\n    for base in (AWS_BASE, NOMADS_BASE):"
)
text = text.replace(
    "                return cycle\n        except (httpx.RequestError, GfsFetchError) as e:", 
    save_code + "        except (httpx.RequestError, GfsFetchError) as e:"
)

with open('src/inference/gfs_live.py', 'w', encoding='utf-8') as f:
    f.write(text)

