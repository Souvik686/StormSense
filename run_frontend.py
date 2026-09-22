"""StormSense frontend-only static server (SPLIT MODE).

Serves `frontend/` over HTTP so the dashboard runs independently of the API.
The page then calls the backend cross-origin; `frontend/js/config.js` resolves
that base automatically for the dev ports, and the backend allows them via
CORS_ALLOW_ORIGINS in backend/main.py.

    Terminal 1:  python run_backend.py      # API on :8000
    Terminal 2:  python run_frontend.py     # dashboard on :3000

Merged single-command mode is unchanged and still available:

    python run_server.py                    # dashboard + API on :8000

Environment:
    FRONTEND_HOST   default 127.0.0.1
    FRONTEND_PORT   default 3000
    API_BASE_URL    injected into the page so the frontend targets a specific
                    backend (default: http://<host>:8000)

This intentionally uses only the standard library -- the frontend is plain
static HTML/CSS/JS with no build step, and adding a bundler would be a larger
change than the task needs.
"""
import functools
import http.server
import json
import os
import socketserver
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

HOST = os.getenv("FRONTEND_HOST", "127.0.0.1")
PORT = int(os.getenv("FRONTEND_PORT", "3000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{HOST}:8000")


_MAPS_KEYS_CACHE = None


def _maps_keys():
    """Browser Google Maps keys, fetched once from the backend's /api/config.

    Cached because index.html is re-served on every navigation and the keys do
    not change while the backend is up. Returns [] if the backend is not
    reachable yet -- the page then reports "no Google Maps key configured"
    rather than failing to parse, and a reload once the backend is up fixes it.
    """
    global _MAPS_KEYS_CACHE
    if _MAPS_KEYS_CACHE is not None:
        return _MAPS_KEYS_CACHE
    try:
        with urllib.request.urlopen(API_BASE_URL + "/api/config", timeout=10) as r:
            data = json.load(r)
        keys = data.get("google_maps_api_keys") or []
        if not isinstance(keys, list):
            keys = []
        _MAPS_KEYS_CACHE = keys
    except Exception as e:
        sys.stderr.write(f"[frontend] could not fetch maps keys from backend: {e}\n")
        _MAPS_KEYS_CACHE = []
    return _MAPS_KEYS_CACHE


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static handler that injects the API base into index.html.

    Injecting at serve time (rather than editing the file) keeps the same
    checked-in index.html working in merged mode, where the API is same-origin
    and no absolute base should be forced.
    """

    def end_headers(self):
        # A dev convenience only: the dashboard fetches fresh model output on
        # every refresh, and a cached app.js silently hides code changes.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._serve_index()
        return super().do_GET()

    def _serve_index(self):
        index = os.path.join(FRONTEND_DIR, "index.html")
        try:
            with open(index, "r", encoding="utf-8") as f:
                html = f.read()
        except OSError as e:
            self.send_error(500, f"cannot read index.html: {e}")
            return

        # The Google Maps key list is substituted server-side in merged mode by
        # serve_dashboard(). Split mode serves the raw file, so the same
        # substitution must happen here or the page throws a ReferenceError on
        # the placeholder token and the map never initialises. Keys come from
        # the backend's browser-safe /api/config (never from .env directly --
        # that file also holds server-only secrets).
        html = html.replace(
            "GOOGLE_MAPS_API_KEYS_JSON_PLACEHOLDER", json.dumps(_maps_keys())
        )
        keys = _maps_keys()
        html = html.replace("GOOGLE_MAPS_API_KEY_PLACEHOLDER", keys[0] if keys else "")

        # config.js reads window.STORMSENSE_API_BASE first, so this wins over
        # every other resolution path without touching the file on disk.
        inject = (
            "<script>window.STORMSENSE_API_BASE = "
            f'"{API_BASE_URL}";</script>\n'
        )
        marker = '<script src="js/config.js"></script>'
        if marker in html:
            html = html.replace(marker, inject + marker, 1)
        else:
            html = html.replace("</head>", inject + "</head>", 1)

        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("[frontend] " + (fmt % args) + "\n")


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    if not os.path.isdir(FRONTEND_DIR):
        raise SystemExit(f"frontend directory not found: {FRONTEND_DIR}")

    handler = functools.partial(Handler, directory=FRONTEND_DIR)
    print("=" * 68)
    print("  StormSense FRONTEND (split mode)")
    print(f"  Dashboard : http://{HOST}:{PORT}/index.html")
    print(f"  API target: {API_BASE_URL}")
    print("  Start the backend separately:  python run_backend.py")
    print("=" * 68)
    with ReusableTCPServer((HOST, PORT), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[frontend] stopped")


if __name__ == "__main__":
    main()
