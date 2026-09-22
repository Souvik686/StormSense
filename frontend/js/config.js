/**
 * StormSense frontend runtime configuration.
 *
 * Resolves the backend API base URL for BOTH supported deployment modes:
 *
 *   MERGED  (python run_server.py)  -- FastAPI serves this page AND the API
 *                                      from the same origin, so the base is ""
 *                                      and every request stays same-origin.
 *
 *   SPLIT   (frontend :3000, API :8000) -- the page is served by a static
 *                                      frontend server and must call the API
 *                                      cross-origin, so an absolute base is
 *                                      required and the backend must allow
 *                                      this page's origin via CORS.
 *
 * Resolution order (first match wins), so a deployment can override without
 * editing code:
 *
 *   1. window.STORMSENSE_API_BASE  -- set by an inline <script> or injected
 *                                     by a deploy step / container env.
 *   2. <meta name="stormsense-api-base" content="...">  -- static hosting.
 *   3. ?api=<url> query parameter   -- ad-hoc testing against another backend.
 *   4. Same origin ("")             -- correct whenever the API is served by
 *                                     the same host that served this page.
 *   5. Dev fallback                 -- only when the page is on a known
 *                                     frontend-only dev port, point at the
 *                                     conventional local backend port.
 *
 * This replaces a hardcoded `http://127.0.0.1:8000` with port-sniffing, which
 * broke any deployment not on those exact ports and forced cross-origin
 * requests even when the API was same-origin.
 */
(function () {
  "use strict";

  // Ports used by a frontend-only dev server. A page served from one of these
  // cannot be same-origin with the API, so it needs an absolute base.
  var FRONTEND_DEV_PORTS = ["3000", "5173", "8080"];
  var DEFAULT_DEV_API_PORT = "8000";

  function trimTrailingSlash(u) {
    return (typeof u === "string" && u.length > 1 && u.charAt(u.length - 1) === "/")
      ? u.slice(0, -1)
      : u;
  }

  function fromMeta() {
    try {
      var el = document.querySelector('meta[name="stormsense-api-base"]');
      var v = el && el.getAttribute("content");
      return (v && v.trim()) ? v.trim() : null;
    } catch (e) { return null; }
  }

  function fromQuery() {
    try {
      var v = new URLSearchParams(window.location.search).get("api");
      if (!v) return null;
      v = v.trim();
      // Only accept absolute http(s) origins, so a crafted link cannot point
      // the dashboard at an arbitrary scheme.
      return /^https?:\/\//i.test(v) ? v : null;
    } catch (e) { return null; }
  }

  function resolve() {
    if (typeof window.STORMSENSE_API_BASE === "string") {
      return trimTrailingSlash(window.STORMSENSE_API_BASE);
    }
    var meta = fromMeta();
    if (meta !== null) return trimTrailingSlash(meta);

    var q = fromQuery();
    if (q !== null) return trimTrailingSlash(q);

    // file:// has no usable origin, so fall back to the dev backend.
    if (window.location.protocol === "file:") {
      return "http://127.0.0.1:" + DEFAULT_DEV_API_PORT;
    }

    if (FRONTEND_DEV_PORTS.indexOf(window.location.port) !== -1) {
      return window.location.protocol + "//" + window.location.hostname +
             ":" + DEFAULT_DEV_API_PORT;
    }

    // Same origin: merged mode, or any reverse-proxied deployment.
    return "";
  }

  var base = resolve();

  window.StormSenseConfig = {
    apiBase: base,
    // "" means same-origin; anything else is a cross-origin (split) setup.
    isSplitMode: base !== "",
    url: function (path) {
      if (!path) return base || "/";
      return base + (path.charAt(0) === "/" ? path : "/" + path);
    }
  };
})();
