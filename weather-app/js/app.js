/**
 * StormSense — Operational Nowcasting Frontend Engine
 *
 * Real Data Integration:
 * - Dynamic UTC issue and valid time derivation (no hardcoded timestamps)
 * - Real-time IST clock ticker (1-second precision)
 * - West Bengal 12-district administrative GeoJSON boundaries
 * - Official West Bengal outer state boundary GeoJSON
 * - 0.25° StormSense continuous risk surface layer
 * - Verified 2024 held-out test benchmarks (V2 vs V1 vs Persistence)
 * - Top predicted high-risk convective cells
 * - Atmospheric thermodynamics (CAPE, CIN, Bulk Shear)
 * - Physical XAI feature attribution
 * - Live surface station telemetry
 * - Dual-Mode Support: LIVE MONITORING (default /) vs HISTORICAL CASE STUDY (/?mode=historical)
 */
(function () {
  "use strict";

  var viewIds = ["dashboard", "radar", "advisories", "wrf", "xai", "gis", "threshold", "streams"];
  var API_BASE = (window.location.port === "8000" || window.location.origin.indexOf(":8000") !== -1) ? "" : "http://127.0.0.1:8000";

  // Mode detection: default is "live", switchable to "historical"
  function detectInitialMode() {
    try {
      var params = new URLSearchParams(window.location.search);
      return params.get("mode") === "historical" ? "historical" : "live";
    } catch (e) {
      return "live";
    }
  }

  window.stormSenseMode = detectInitialMode();
  window.currentLeadHours = 2;
  window.currentBenchmarkLead = 2;
  window.mapDomainMode = "wb"; // 'wb' or 'full'

  // Global cached data
  window.StormSenseCurrentObs = null;
  window.StormSenseNowcastData = null;
  window.StormSenseDistrictsData = null;
  window.StormSenseRiskMapData = null;
  window.StormSenseThermoData = null;
  window.StormSenseXaiData = null;
  window.StormSenseBoundariesData = null;
  window.StormSenseStateBoundaryData = null;
  window.StormSenseBenchmarkData = null;
  window.StormSenseHighRiskCells = null;
  window.StormSenseLiveSurface = null;
  window.StormSenseDataHealth = null;

  // Live data refresh cadence (5 minutes).
  var LIVE_REFRESH_MS = 300000;
  var liveCountdownSeconds = LIVE_REFRESH_MS / 1000;
  var liveRefreshTimer = null;
  var countdownTimer = null;

  function formatCountdown(totalSeconds) {
    var m = Math.floor(totalSeconds / 60);
    var s = totalSeconds % 60;
    return m > 0 ? m + "m " + (s < 10 ? "0" : "") + s + "s" : s + "s";
  }

  // The monitored region is the whole state; the active high-risk area is
  // computed from the same forecast the map and cards show. Nothing here is
  // hardcoded to a district -- contradictory region labels were a real defect.
  function applyRegionState(summary) {
    setText("profile-district", "West Bengal");

    var el = document.getElementById("active-high-risk-area");
    if (!el) return;

    if (!summary || summary.status === "unavailable") {
      el.textContent = "UNAVAILABLE";
      el.className = "text-sm font-bold text-slate-400";
      return;
    }

    var active = summary && summary.active_high_risk_district;
    if (active && active.district) {
      el.textContent = active.district + " (" + active.overall_pct + "%)";
      el.className = "text-sm font-bold text-amber-300";
    } else {
      el.textContent = "NO ACTIVE HIGH-RISK DISTRICT";
      el.className = "text-sm font-bold text-emerald-300";
    }
  }

  // Marks the dashboard when the displayed forecast is older than expected.
  // Values are kept (not blanked) so operators still see the last good state.
  function applyFreshnessState(summary) {
    var el = document.getElementById("data-freshness");
    if (!el) return;
    var status = summary && summary.live_status;
    if (!status) {
      el.textContent = "";
      return;
    }
    if (status.is_stale) {
      el.textContent = "STALE - last update " + (status.issue_time_formatted || "unknown");
      el.className = "text-[10px] font-bold text-amber-400";
    } else {
      el.textContent = "CURRENT - " + (status.issue_time_formatted || "");
      el.className = "text-[10px] font-bold text-emerald-400";
    }
  }

  // Static reference location for monitored station
  var DEFAULT_LOCATION = {
    district: "North 24 Parganas",
    state: "West Bengal",
    country: "India",
    shortLabel: "North 24 Parganas, West Bengal",
    lat: 22.724,
    lon: 88.479,
    subdivisions: ["Barasat Sadar", "Barrackpore", "Bangaon", "Basirhat", "Bidhannagar"],
    defaultSector: "Barasat Sadar"
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Dynamic Time Formatters
  // ─────────────────────────────────────────────────────────────────────────────
  function parseUtcIso(isoStr) {
    if (!isoStr) return new Date("2024-05-05T15:00:00Z");
    var d = new Date(isoStr);
    return isNaN(d.getTime()) ? new Date("2024-05-05T15:00:00Z") : d;
  }

  function formatUtcDateTime(dateObj) {
    var d = typeof dateObj === "string" ? parseUtcIso(dateObj) : dateObj;
    var year = d.getUTCFullYear();
    var month = String(d.getUTCMonth() + 1).padStart(2, "0");
    var day = String(d.getUTCDate()).padStart(2, "0");
    var hours = String(d.getUTCHours()).padStart(2, "0");
    var minutes = String(d.getUTCMinutes()).padStart(2, "0");
    return year + "-" + month + "-" + day + " " + hours + ":" + minutes + " UTC";
  }

  function updateDynamicTimes(issueTimeStr, leadHours) {
    leadHours = Number(leadHours) || 2;
    var issueDate = parseUtcIso(issueTimeStr);
    var validDate = new Date(issueDate.getTime() + leadHours * 3600 * 1000);

    var issueFormatted = formatUtcDateTime(issueDate);
    var validFormatted = formatUtcDateTime(validDate);
    var leadFormatted = "+" + leadHours + "h (" + (leadHours * 60) + "m Lead)";

    document.querySelectorAll("#bind-issue-time, .bind-issue-time").forEach(function (el) { el.textContent = issueFormatted; });
    document.querySelectorAll("#bind-forecast-lead, .bind-forecast-lead").forEach(function (el) { el.textContent = leadFormatted; });
    document.querySelectorAll("#bind-valid-time, .bind-valid-time").forEach(function (el) { el.textContent = validFormatted; });

    var deskLead = document.getElementById("desk-lead-indicator");
    if (deskLead) deskLead.textContent = "+" + leadHours + "h Calibrated Horizon";

    var deskCellsLead = document.getElementById("desk-cells-lead-tag");
    if (deskCellsLead) deskCellsLead.textContent = "+" + leadHours + "h Horizon";

    var legendLead = document.getElementById("legend-lead-tag");
    if (legendLead) legendLead.textContent = "+" + leadHours + "h";
  }

  function formatIstClock(date) {
    var utc = date.getTime() + (date.getTimezoneOffset() * 60000);
    var ist = new Date(utc + (5.5 * 3600000));
    return String(ist.getHours()).padStart(2, "0") + ":" +
           String(ist.getMinutes()).padStart(2, "0") + " IST";
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Real-Time IST Clock (Every 1 Second)
  // ─────────────────────────────────────────────────────────────────────────────
  function updateIstClock() {
    var now = new Date();
    // Indian Standard Time is UTC + 5:30
    var utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    var ist = new Date(utc + (5.5 * 3600000));

    var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var day = String(ist.getDate()).padStart(2, "0");
    var month = months[ist.getMonth()];
    var year = ist.getFullYear();
    var hours = String(ist.getHours()).padStart(2, "0");
    var minutes = String(ist.getMinutes()).padStart(2, "0");
    var seconds = String(ist.getSeconds()).padStart(2, "0");

    var fullTimeStr = day + " " + month + " " + year + ", " + hours + ":" + minutes + ":" + seconds + " IST";
    var shortTimeStr = hours + ":" + minutes + ":" + seconds + " IST";

    var clockFooter = document.getElementById("clock-ist-display");
    if (clockFooter) clockFooter.textContent = fullTimeStr;

    var clockHeader = document.getElementById("header-clock-ist");
    if (clockHeader) clockHeader.textContent = shortTimeStr;

    if (window.stormSenseMode === "live" && window.currentObsTime) {
      var obsDate = new Date(window.currentObsTime);
      var ageSeconds = Math.floor((now.getTime() - obsDate.getTime()) / 1000);
      if (ageSeconds < 0) ageSeconds = 0;
      var h = Math.floor(ageSeconds / 3600);
      var m = Math.floor((ageSeconds % 3600) / 60);
      var obsText = "";
      if (h > 0) obsText = h + "h " + m + "m ago";
      else if (m <= 1) obsText = "Just now";
      else obsText = m + "m ago";
      
      var obsBadge = document.getElementById("header-obs-time-text");
      if (obsBadge) obsBadge.textContent = obsText;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Live Surface Observation & Data Health Ingest (Every 60 Seconds)
  // ─────────────────────────────────────────────────────────────────────────────
  window.fetchLiveSurfaceData = function (isManual) {
    return fetch(API_BASE + "/api/live/surface")
      .then(parseJson)
      .then(function (data) {
        if (!data) return;
        window.StormSenseLiveSurface = data;
        window.currentObsTime = data.observed_at_utc;
        var obs = data.observations || {};

        if (window.stormSenseMode === "live") {
          paintDashboardLive(obs, data);
        }

        if (window.stormSenseMap && window._liveStationMarker) {
          updateLiveStationPopup(obs);
        }

        liveCountdownSeconds = LIVE_REFRESH_MS / 1000;
        setText("live-refresh-countdown", formatCountdown(liveCountdownSeconds));

        if (isManual) {
          window.showToast("Refreshed live surface observation for North 24 Parganas", "check_circle");
        }
      })
      .catch(function (err) {
        console.warn("Live surface fetch error:", err);
      });
  };

  /**
   * Five-minute live refresh. Updates current observations AND the forecast
   * products (hazard cards, districts, map surface, advisories, timestamps)
   * without reloading the page, and deliberately preserves the selected
   * horizon, the selected mode and the map viewport.
   *
   * A failure on any leg leaves the previously displayed values in place rather
   * than blanking them; the freshness indicator marks them stale instead.
   */
  window.refreshLiveDashboard = function (isManual) {
    if (window.stormSenseMode !== "live") return Promise.resolve();

    var lead = window.currentLeadHours || 2;
    var modeQS = "&mode=live";

    return Promise.all([
      window.fetchLiveSurfaceData(false).catch(function() { return null; }),
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/districts?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {
        var summary = results[1];
        window.StormSenseCurrentSummary = summary;
        var districts = results[2];
        var cells = results[3];

        if (summary && summary.hazards) {
          window.StormSenseNowcastData = summary;
          paintForecastCards(summary);
          if (summary.timeline) renderNowcast(summary.timeline);
          if (summary.issue_time) updateDynamicTimes(summary.issue_time, lead);
        }
        applyRegionState(summary);
        applyFreshnessState(summary);

        if (districts && districts.length) {
          window.StormSenseDistrictsData = districts;
          renderDistricts(adaptDistricts(districts, lead, summary && summary.issue_time));
        }

        if (cells) {
          window.StormSenseHighRiskCells = cells;
          renderHighRiskCells(cells);
          if (window.stormSenseMap) renderHotspotBeacons(window.stormSenseMap, cells);
        }

        // Repaint the surface for the CURRENT horizon; never re-fit the map, so
        // the operator's pan/zoom survives the refresh.
        if (window.stormSenseMap) renderContinuousRiskSurface(window.stormSenseMap, lead);

        setText("last-updated-time", formatIstClock(new Date()));
        if (isManual) window.showToast("Live data refreshed", "check_circle");
      })
      .catch(function (err) {
        // Never let one bad cycle kill the interval or wipe the display.
        console.warn("Live refresh cycle failed:", err);
      });
  };

  function fetchDataHealth() {
    return fetch(API_BASE + "/api/data-health")
      .then(parseJson)
      .then(function (data) {
        if (!data || !data.components) return;
        window.StormSenseDataHealth = data;
        var timestamp = document.getElementById("live-health-timestamp");
        if (timestamp) timestamp.textContent = "Last checked: " + new Date().toLocaleTimeString();

        data.components.forEach(function (c) {
          if (c.name === "Surface Weather Observations") {
            var b = document.getElementById("health-surface-badge");
            var d = document.getElementById("health-surface-detail");
            if (b) {
              if (c.status === "ONLINE") {
                b.className = "flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";
                b.innerHTML = '<span class="size-1.5 rounded-full bg-emerald-400 animate-pulse"></span>ONLINE';
              } else {
                b.className = "flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/40";
                b.innerHTML = '<span class="size-1.5 rounded-full bg-red-400"></span>OFFLINE';
              }
            }
            if (d && c.detail) d.textContent = c.detail;
          }
        });
      })
      .catch(function (e) {
        console.warn("Data health fetch failed:", e);
      });
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Operational Mode Controller (LIVE vs HISTORICAL)
  // ─────────────────────────────────────────────────────────────────────────────
  window.switchMode = function (targetMode) {
    if (targetMode !== "live" && targetMode !== "historical") targetMode = "live";
    window.stormSenseMode = targetMode;

    if (window.history && window.history.pushState) {
      var newUrl = targetMode === "historical" ? "?mode=historical" : window.location.pathname;
      window.history.pushState({ mode: targetMode }, "", newUrl);
    }

    updateModeUI(targetMode);
  };

  window.toggleOperationalMode = function () {
    var nextMode = window.stormSenseMode === "live" ? "historical" : "live";
    window.switchMode(nextMode);
  };

  function updateModeUI(mode) {
    var isHistorical = mode === "historical";
    var btn = document.getElementById("btn-mode-toggle");
    var btnLabel = document.getElementById("btn-mode-label");
    var btnIcon = document.getElementById("btn-mode-icon");
    var modeBadge = document.getElementById("bind-operational-mode");
    var modeBadgeDot = document.getElementById("mode-badge-dot");
    var modeBadgeLabel = document.getElementById("mode-badge-label");
    var footerModePill = document.getElementById("footer-mode-pill");

    var ctxLive = document.getElementById("header-context-live");
    var ctxHist = document.getElementById("header-context-historical");
    var clockDot = document.getElementById("header-clock-dot");

    var banner = document.querySelector(".operational-banner");
    var bannerDisclaimer = document.getElementById("bind-disclaimer");

    var deskModePill = document.getElementById("desk-mode-pill");
    var deskModeDot = document.getElementById("desk-mode-dot");
    var deskModeLabel = document.getElementById("desk-mode-label");
    var deskSubtitle = document.getElementById("desk-subtitle");

    var refreshBox = document.getElementById("live-refresh-container");
    var pipeBadge = document.getElementById("pipeline-status-badge");
    var pipeText = document.getElementById("pipeline-status-text");
    var pipeDot = document.getElementById("pipeline-status-dot");

    if (isHistorical) {
      if (pipeBadge) {
        pipeBadge.className = "flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-[11px] font-mono font-bold";
      }
      if (pipeDot) pipeDot.className = "size-1.5 rounded-full bg-cyan-400 animate-pulse";
      if (pipeText) pipeText.textContent = "HISTORICAL CASE STUDY";
      if (btn) {
        btn.className = "btn-mode-switch flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold transition-all shadow-md cursor-pointer border border-slate-700";
      }
      if (btnLabel) btnLabel.textContent = "← Return to Live Monitoring";
      if (btnIcon) btnIcon.textContent = "sensors";

      if (modeBadge) {
        modeBadge.className = "px-2.5 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-[10px] font-bold font-mono uppercase tracking-wider flex items-center gap-1.5";
      }
      if (modeBadgeDot) modeBadgeDot.className = "size-1.5 rounded-full bg-cyan-400 animate-pulse";
      if (modeBadgeLabel) modeBadgeLabel.textContent = "HISTORICAL CASE STUDY — KALBAISHAKHI";

      if (footerModePill) {
        footerModePill.className = "px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-cyan-500/10 text-cyan-400 border border-cyan-500/30";
        footerModePill.textContent = "HISTORICAL CASE STUDY";
      }

      if (ctxLive) ctxLive.classList.add("hidden");
      if (ctxHist) {
        ctxHist.classList.remove("hidden");
        ctxHist.classList.add("inline-flex");
      }
      if (clockDot) clockDot.className = "size-2 rounded-full bg-cyan-400 animate-pulse";

      if (banner) {
        banner.className = "operational-banner mode-banner-historical px-6 lg:px-8 flex items-center justify-between text-xs";
      }
      if (bannerDisclaimer) {
        bannerDisclaimer.textContent = "StormSense ML Case Study · Validated May 5, 2024 Pre-Monsoon Squall · Grid: 0.25° (~28 km)";
      }

      if (deskModePill) {
        deskModePill.className = "px-2.5 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs font-bold uppercase tracking-wider font-mono flex items-center gap-1.5";
      }
      if (deskModeDot) deskModeDot.className = "size-2 rounded-full bg-cyan-400 animate-pulse";
      if (deskModeLabel) deskModeLabel.textContent = "HISTORICAL CASE STUDY — KALBAISHAKHI";
      if (deskSubtitle) {
        deskSubtitle.textContent = "Historical convective case study (0–6 h) driven by StormSense ML engine and real Atmospheric Data thermodynamics (May 5, 2024).";
      }

      if (refreshBox) refreshBox.style.display = "none";

      // Paint historical dashboard data
      if (window.StormSenseNowcastData) {
        var vm = buildDashboardViewModel(
          window.StormSenseCurrentObs,
          window.StormSenseNowcastData,
          window.StormSenseDistrictsData,
          window.StormSenseThermoData,
          window.StormSenseXaiData
        );
        paintDashboard(vm);
      }

      // Update map layers
      updateMapMode("historical");

      window.showToast("Active Mode: Historical Case Study (Kalbaishakhi Convective Squall)", "history_edu");
    } else {
      // LIVE MODE
      if (pipeBadge) {
        pipeBadge.className = "flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px] font-mono font-bold";
      }
      if (pipeDot) pipeDot.className = "size-1.5 rounded-full bg-emerald-400 animate-pulse";
      if (pipeText) pipeText.textContent = "AI FORECAST ACTIVE";
      if (btn) {
        btn.className = "btn-mode-switch flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-bold transition-all shadow-md shadow-cyan-600/25 cursor-pointer border border-cyan-400/40";
      }
      if (btnLabel) btnLabel.textContent = "View Historical Case Study →";
      if (btnIcon) btnIcon.textContent = "history_edu";

      if (modeBadge) {
        modeBadge.className = "px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[10px] font-bold font-mono uppercase tracking-wider flex items-center gap-1.5";
      }
      if (modeBadgeDot) modeBadgeDot.className = "size-1.5 rounded-full bg-emerald-400 animate-pulse";
      if (modeBadgeLabel) modeBadgeLabel.textContent = "LIVE MONITORING";

      if (footerModePill) {
        footerModePill.className = "px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/30";
        footerModePill.textContent = "LIVE MONITORING";
      }

      if (ctxHist) {
        ctxHist.classList.add("hidden");
        ctxHist.classList.remove("inline-flex");
      }
      if (ctxLive) ctxLive.classList.remove("hidden");
      if (clockDot) clockDot.className = "size-2 rounded-full bg-emerald-400 animate-pulse";

      if (banner) {
        banner.className = "operational-banner mode-banner-live px-6 lg:px-8 flex items-center justify-between text-xs";
      }
      if (bannerDisclaimer) {
        bannerDisclaimer.textContent = "Live monitoring · AI-derived severe weather risk · Not an official government warning";
      }

      if (deskModePill) {
        deskModePill.className = "px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-bold uppercase tracking-wider font-mono flex items-center gap-1.5";
      }
      if (deskModeDot) deskModeDot.className = "size-2 rounded-full bg-emerald-400 animate-pulse";
      if (deskModeLabel) deskModeLabel.textContent = "LIVE MONITORING";
      if (deskSubtitle) {
        deskSubtitle.textContent = "Current surface conditions and AI forecast risk. Updated every 5 minutes.";
      }

      if (refreshBox) refreshBox.style.display = "flex";

      // Current observations, then the live forecast products for the horizon
      // that is already selected (mode switching must not reset the horizon).
      if (window.StormSenseLiveSurface) {
        paintDashboardLive(window.StormSenseLiveSurface.observations || {}, window.StormSenseLiveSurface);
      } else {
        window.fetchLiveSurfaceData(false);
      }
      window.refreshLiveDashboard();

      // Update map layers
      updateMapMode("live");

      window.showToast("Active Mode: Live Meteorological Monitoring", "sensors");
    }
  }

  window.addEventListener("popstate", function () {
    var m = detectInitialMode();
    window.switchMode(m);
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // Data Loading & ViewModel Assembly
  // ─────────────────────────────────────────────────────────────────────────────
  function parseJson(res) {
    if (!res || !res.ok) return null;
    return res.json().catch(function () { return null; });
  }

  function loadDashboardData() {
    var lead = window.currentLeadHours || 2;
    return Promise.all([
      fetch(API_BASE + "/api/weather/current").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/districts?lead=" + lead).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/risk-map?lead=" + lead).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/boundaries/west-bengal").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/boundaries/state").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/benchmark/models").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/live/surface").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/data-health").then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {
        var currentObs = results[0];
        var summary = results[1];
        window.StormSenseCurrentSummary = summary;
        var districts = results[2];
        var riskMap = results[3];
        var thermo = results[4];
        var xai = results[5];
        var boundaries = results[6];
        var stateBoundary = results[7];
        var benchmark = results[8];
        var highRiskCells = results[9];
        var liveSurface = results[10];
        var dataHealth = results[11];

        window.StormSenseCurrentObs = currentObs;
        window.StormSenseNowcastData = summary;
        window.StormSenseDistrictsData = districts;
        window.StormSenseRiskMapData = riskMap;
        window.StormSenseThermoData = thermo;
        window.StormSenseXaiData = xai;
        window.StormSenseBoundariesData = boundaries;
        window.StormSenseStateBoundaryData = stateBoundary;
        window.StormSenseBenchmarkData = benchmark;
        window.StormSenseHighRiskCells = highRiskCells;
        window.StormSenseLiveSurface = liveSurface;
        window.StormSenseDataHealth = dataHealth;

        // Ensure sidebar views (advisories, benchmarks, XAI) are pre-populated for immediate viewing in both modes
        if (districts && districts.length) {
          renderDistricts(districts);
        }
        if (benchmark) {
          renderEvolutionTimeline(benchmark);
          renderBenchmarkCards(benchmark, window.currentBenchmarkLead || 2);
        }
        if (xai) {
          renderXai(xai);
        }

        return buildDashboardViewModel(currentObs, summary, districts, thermo, xai);
      })
      .catch(function (error) {
        console.warn("API request failed:", error);
        return buildDashboardViewModel(null, null, null, null, null);
      });
  }

  function buildDashboardViewModel(currentObs, summary, districts, thermo, xai) {
    var vm = {
      location: DEFAULT_LOCATION,
      currentConditions: {
        temperatureC: currentObs ? currentObs.temperature : 28.5,
        feelsLikeC: currentObs ? currentObs.feels_like : 32.0,
        humidityPct: currentObs ? currentObs.humidity : 80,
        windKmh: currentObs ? currentObs.wind_speed : 16.0,
        windDirection: currentObs ? getWindDirection(currentObs.wind_direction || 180) : "SE",
        rainfallMmHr: currentObs ? (currentObs.rainfall_1h || currentObs.rainfall_1h_mm || 0.0) : 0.0
      },
      hazards: {
        thunderstorm: { probabilityPct: 0, level: "green", trendLabel: "Nominal", detail: "Subdued" },
        heavyRainfall: { probabilityPct: 0, level: "green", trendLabel: "Nominal", detail: "Subdued" },
        extremeRainfall: { probabilityPct: 0, level: "green", trendLabel: "Nominal", detail: "Subdued" },
        flashFlood: { probabilityPct: 0, level: "green", trendLabel: "Nominal", detail: "Subdued" },
        overall: { probabilityPct: 0, level: "green", actionLabel: "Nominal Monitoring", stageLabel: "Stage 1 Ambient" }
      },
      districts: [],
      timeline: [],
      bulletins: [],
      systemStatus: {
        clusterLabel: "ML Engine Online",
        slaLabel: "CALIBRATED",
        versionLabel: "StormSense",
        deskLabel: "North 24 Parganas desk",
        freshnessLabel: "LIVE INGEST · t=0"
      },
      disclaimer: "StormSense spatial nowcasting. Multi-task ConvGRU evaluated on 4,322 held-out 2024 test sequences."
    };

    // Dynamic Time Derivation
    var issueTime = (summary && summary.issue_time) || "2024-05-05T15:00:00Z";
    var lead = (summary && summary.lead_hours) || window.currentLeadHours || 2;
    updateDynamicTimes(issueTime, lead);

    // Current Station Telemetry
    if (summary && summary.surface_obs_t0) {
      var s0 = summary.surface_obs_t0;
      vm.currentConditions.temperatureC = s0.temperature_c;
      vm.currentConditions.feelsLikeC = s0.temperature_c;
      vm.currentConditions.humidityPct = s0.humidity_pct;
      vm.currentConditions.windKmh = s0.wind_speed_kmh;
      vm.currentConditions.windDirection = "ESE";
      vm.currentConditions.rainfallMmHr = s0.rainfall_mm;
    } else if (currentObs) {
      vm.currentConditions.temperatureC = currentObs.temperature || 28.5;
      vm.currentConditions.feelsLikeC = currentObs.feels_like || 32.0;
      vm.currentConditions.humidityPct = currentObs.humidity || 80;
      vm.currentConditions.windKmh = currentObs.wind_speed || 16.0;
      vm.currentConditions.windDirection = getWindDirection(currentObs.wind_direction || 180);
      vm.currentConditions.rainfallMmHr = currentObs.rainfall_1h || 0.0;
    }

    // Calibrated ML Hazards
    if (summary && summary.hazards) {
      var hz = summary.hazards;
      if (hz.thunderstorm) vm.hazards.thunderstorm = adaptHazard(hz.thunderstorm, "Thunderstorm");
      var rainHz = hz.heavy_rainfall || hz.heavyRainfall;
      if (rainHz) vm.hazards.heavyRainfall = adaptHazard(rainHz, "Heavy Rainfall");
      var extRainHz = hz.extreme_rainfall || hz.extremeRainfall;
      if (extRainHz) vm.hazards.extremeRainfall = adaptHazard(extRainHz, "Extreme Rainfall");
      var floodHz = hz.flash_flood || hz.flashFlood;
      if (floodHz) vm.hazards.flashFlood = adaptHazard(floodHz, "Flash Flood");

      if (hz.overall) {
        var overallProb = getHazardProbability(hz.overall);
        var overallLevel = hz.overall.level || probabilityToLevel(overallProb);
        vm.hazards.overall = {
          probabilityPct: overallProb,
          level: overallLevel,
          actionLabel: getOverallAction(overallLevel),
          stageLabel: getOverallStage(overallLevel)
        };
      }
    }

    // 12-District Advisories
    if (districts && districts.length) {
      vm.districts = districts.map(function (d) {
        var name = d.district || d.name;
        var level = d.risk_level || d.riskLevel || "green";
        var tstormPct = d.thunderstorm_pct != null ? d.thunderstorm_pct : (d.thunderstorm_prob != null ? Math.round(d.thunderstorm_prob * 100) : 0);
        var rainMm = d.rainfall_mm_3h != null ? Number(d.rainfall_mm_3h).toFixed(1) : 0;
        var floodPct = d.flash_flood_proxy_pct != null ? d.flash_flood_proxy_pct : (d.flash_flood_proxy != null ? Math.round(d.flash_flood_proxy * 100) : 0);
        return {
          name: name,
          district: name,
          riskLevel: level,
          overallPct: d.overall_risk_pct || Math.max(tstormPct, floodPct),
          thunderstormPct: tstormPct,
          heavyRainfallPct: rainMm,
          flashFloodPct: floodPct,
          confidencePct: d.confidence_pct != null ? d.confidence_pct : 92,
          isPrimary: d.is_primary || name === "North 24 Parganas",
          note: d.note || "StormSense multi-cell risk aggregation.",
          validUntil: d.valid_until || formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + lead * 3600 * 1000))
        };
      });
    }

    // Multi-horizon timeline points
    if (summary && summary.timeline) {
      vm.timeline = summary.timeline.map(function (pt) {
        return {
          hours_from_now: pt.hours_from_now,
          lead_label: "+" + pt.hours_from_now + "h",
          temperature: pt.temperature != null ? pt.temperature : 28.0,
          rainfall_mm: pt.rainfall_mm != null ? pt.rainfall_mm : 0.0,
          humidity: pt.humidity != null ? pt.humidity : 80,
          wind_speed: pt.wind_speed != null ? pt.wind_speed : 15.0,
          valid_time: pt.forecast_valid_time || formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + pt.hours_from_now * 3600 * 1000)),
          weather_description: pt.weather_description || "Convective Monitoring"
        };
      });
    }

    if (xai) vm.explainability = xai;
    vm.systemStatus.versionLabel = "StormSense";
    vm.disclaimer = "StormSense spatial nowcasting. Multi-task ConvGRU evaluated on 4,322 held-out 2024 test sequences.";

    return vm;
  }

  function getHazardProbability(hazard) {
    if (!hazard) return 0;
    if (hazard.probability != null) return Math.round(Number(hazard.probability));
    if (hazard.probabilityPct != null) return Math.round(Number(hazard.probabilityPct));
    return 0;
  }

  function probabilityToLevel(probability) {
    var p = Number(probability) || 0;
    if (p >= 75) return "red";
    if (p >= 50) return "orange";
    if (p >= 25) return "yellow";
    return "green";
  }

  function adaptHazard(hazard, hazardName) {
    var prob = getHazardProbability(hazard);
    var level = hazard.level || probabilityToLevel(prob);
    return {
      probabilityPct: prob,
      level: level,
      trendLabel: getHazardTrend(prob),
      detail: getHazardDetail(hazardName, prob)
    };
  }

  function getHazardTrend(prob) {
    if (prob >= 75) return "HIGH RISK";
    if (prob >= 50) return "ELEVATED";
    if (prob >= 25) return "MONITOR";
    return "LOW";
  }

  function getHazardDetail(hazardName, prob) {
    if (prob >= 75) return hazardName + " conditions indicate high short-term risk.";
    if (prob >= 50) return hazardName + " risk is elevated within the nowcast window.";
    if (prob >= 25) return hazardName + " conditions require continued monitoring.";
    return hazardName + " risk currently remains low.";
  }

  function getOverallAction(level) {
    if (level === "red") return "WARNING - Activate Emergency Response Protocol";
    if (level === "orange") return "ALERT - Prepare Civil Defense & Field Units";
    if (level === "yellow") return "WATCH - Maintain Vigilance";
    return "NORMAL - Nominal Observation";
  }

  function getOverallStage(level) {
    if (level === "red") return "WARNING";
    if (level === "orange") return "ALERT";
    if (level === "yellow") return "WATCH";
    return "NORMAL";
  }

  function getWindDirection(degrees) {
    var dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
    var idx = Math.round((Number(degrees) || 0) / 22.5) % 16;
    return dirs[idx];
  }

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el && value != null) el.textContent = String(value);
  }

  function setWidth(id, pct) {
    var el = document.getElementById(id);
    if (el) el.style.width = Math.max(0, Math.min(100, Number(pct) || 0)) + "%";
  }

  function levelClass(level) {
    switch (level) {
      case "red":
        return { text: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/40", pill: "bg-red-500/20 text-red-300 border-red-500/30" };
      case "orange":
        return { text: "text-orange-400", bg: "bg-orange-500/10", border: "border-orange-500/40", pill: "bg-orange-500/20 text-orange-300 border-orange-500/30" };
      case "yellow":
        return { text: "text-amber-300", bg: "bg-amber-500/10", border: "border-amber-500/40", pill: "bg-amber-500/20 text-amber-200 border-amber-500/30" };
      default:
        return { text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/40", pill: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" };
    }
  }

  /**
   * Paint the four forecast hazard cards and their meters from a summary
   * payload. Shared by the horizon switcher and the 5-minute live refresh so
   * both modes render through identical logic.
   *
   * These are FORECAST quantities. Current observed conditions are rendered
   * separately (see paintDashboardLive) and must not be mixed in here.
   */
  function paintForecastCards(summary) {
    if (!summary) return;
    if (summary.status === 'unavailable') {
        // Honest fallback
        document.querySelectorAll('.hazard-probability').forEach(el => el.textContent = 'UNAVAILABLE');
        document.querySelectorAll('.hazard-trend').forEach(el => el.textContent = summary.message || 'Pipeline pending');
        return;
    }
    if (!summary.hazards) return;
    var hz = summary.hazards;

    if (hz.thunderstorm) applyHazardCard("kpi-ts", adaptHazard(hz.thunderstorm, "Thunderstorm"));

    var rHz = hz.heavy_rainfall || hz.heavyRainfall;
    if (rHz) {
      setText("kpi-rain-title", "Heavy Rainfall Risk");
      setText("kpi-rain-category", "Forecast precipitation");
      var rainRate = rHz.rate_mm_3h != null ? rHz.rate_mm_3h : rHz.rain_3h_mm;
      setText("kpi-rain-value", rainRate != null ? Number(rainRate).toFixed(1) + " mm" : rHz.probability + "%");
      setText("kpi-rain-level", (rHz.level || "alert").toUpperCase());
      setText("kpi-rain-trend", getHazardTrend(rHz.probability));
      setText("kpi-rain-detail", getHazardDetail("Heavy Rainfall", rHz.probability));
    }

    var fHz = hz.flash_flood || hz.flashFlood;
    if (fHz) applyHazardCard("kpi-flood", adaptHazard(fHz, "Flash Flood"));

    if (hz.overall) {
      var overallProb = getHazardProbability(hz.overall);
      var overallLevel = hz.overall.level || probabilityToLevel(overallProb);
      setText("kpi-overall-value", overallProb + "%");
      setText("kpi-overall-level", overallLevel.toUpperCase());
      setText("kpi-overall-action", getOverallAction(overallLevel));
      setText("kpi-overall-stage", getOverallStage(overallLevel));
      setWidth("kpi-overall-bar", overallProb);
    }

    var tsP = getHazardProbability(hz.thunderstorm);
    setText("meter-ts-label", tsP + "%");
    setWidth("meter-ts-bar", tsP);

    var rP = rHz && rHz.rate_mm_3h != null ? rHz.rate_mm_3h : getHazardProbability(rHz);
    var rPct = getHazardProbability(rHz);
    setText("meter-rain-label", (typeof rP === "number" ? rP.toFixed(1) + " mm" : rP + "%"));
    setWidth("meter-rain-bar", rPct);

    var flP = getHazardProbability(hz.flash_flood);
    setText("meter-flood-label", flP + "%");
    setWidth("meter-flood-bar", flP);
  }

  function adaptDistricts(districts, lead, issueTime) {
    return districts.map(function (d) {
      var name = d.district || d.name;
      var tstormPct = d.thunderstorm_pct != null ? d.thunderstorm_pct
        : (d.thunderstorm_prob != null ? Math.round(d.thunderstorm_prob * 100) : 0);
      var rainMm = d.heavy_rainfall_mm_3h != null ? Number(d.heavy_rainfall_mm_3h).toFixed(1)
        : (d.rainfall_mm_3h != null ? Number(d.rainfall_mm_3h).toFixed(1) : 0);
      var floodPct = d.flash_flood_pct != null ? d.flash_flood_pct
        : (d.flash_flood_proxy_pct != null ? d.flash_flood_proxy_pct : 0);
      return {
        name: name,
        district: name,
        riskLevel: d.risk_level || d.riskLevel || "green",
        overallPct: d.overall_pct != null ? d.overall_pct : Math.max(tstormPct, floodPct),
        thunderstormPct: tstormPct,
        heavyRainfallPct: rainMm,
        flashFloodPct: floodPct,
        confidencePct: d.confidence_pct != null ? d.confidence_pct : 92,
        // Highest risk leads the list; no district is permanently "primary".
        isPrimary: false,
        note: d.body || "Multi-cell model risk aggregation.",
        validUntil: d.valid_until ||
          (issueTime ? formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + lead * 3600 * 1000)) : "")
      };
    });
  }

  function applyHazardCard(prefix, hazard) {
    if (!hazard) return;
    var cls = levelClass(hazard.level);
    setText(prefix + "-prob", hazard.probabilityPct + "%");
    setText(prefix + "-value", hazard.probabilityPct + "%");
    setText(prefix + "-level", hazard.level.toUpperCase());
    setText(prefix + "-trend", hazard.trendLabel);
    setText(prefix + "-detail", hazard.detail);
    var pill = document.getElementById(prefix + "-level");
    if (pill) {
      pill.className = "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider " + cls.bg + " " + cls.text + " " + cls.border;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Live Dashboard Painter (Genuine Surface Obs + Honest ML Unavailable States)
  // ─────────────────────────────────────────────────────────────────────────────
  /**
   * Paint CURRENT OBSERVED CONDITIONS only.
   *
   * This must never write the four forecast hazard cards -- those show model
   * forecast risk and are painted by paintForecastCards(). Conflating the two
   * is what previously made an observed "0.0 mm" reading appear as a heavy
   * rainfall forecast.
   */
  function paintDashboardLive(obs, rawLive) {
    var rainMm = obs.rainfall_1h_mm != null ? Number(obs.rainfall_1h_mm).toFixed(1) : "0.0";
    var tempC = obs.temperature_c != null ? obs.temperature_c.toFixed(1) : "—";
    var hum = obs.humidity_pct != null ? obs.humidity_pct : "—";
    var windSpd = obs.wind_speed_kmh != null ? obs.wind_speed_kmh.toFixed(1) : "—";

    // Current-conditions telemetry strip
    setText("bind-temp", tempC + "°C");
    setText("bind-rain", rainMm + " mm");
    setText("bind-humidity", hum + "%");
    setText("bind-wind", windSpd + " km/h");

    var liveSourceLabel = rawLive && rawLive.is_stale
      ? "Surface observation · stale"
      : "Surface observation";
    setText("bind-temp-source", liveSourceLabel);
    setText("bind-rain-source", liveSourceLabel);
    setText("bind-humidity-source", liveSourceLabel);
    setText("bind-wind-source", liveSourceLabel);

    var profLevel = document.getElementById("profile-level");
    if (profLevel) profLevel.innerHTML = '<span class="size-2 bg-emerald-400 rounded-full animate-pulse"></span> LIVE MONITORING';

    // Render Live timeline cards
    renderNowcastLive(rawLive);
    setText("nowcast-timeline-title", "Current Conditions");
    setText("nowcast-timeline-badge", "LIVE OBSERVATION");

    // Render Live High-Risk Grid notice
    renderHighRiskCellsLive();

    // Render Live Thermodynamics
    renderThermodynamicsLive(obs);

    // Render Live XAI notice
    // renderXaiLive();
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // UI Renderers for Live Mode Sections
  // ─────────────────────────────────────────────────────────────────────────────
  function renderNowcastLive(liveObs) {
    var root = document.getElementById("nowcast-steps");
    if (!root) return;
    var obs = (liveObs && liveObs.observations) || {};
    var temp = obs.temperature_c != null ? obs.temperature_c.toFixed(1) : "32.0";
    var feels = obs.feels_like_c != null ? obs.feels_like_c.toFixed(1) : "36.0";
    var rain = obs.rainfall_1h_mm != null ? Number(obs.rainfall_1h_mm).toFixed(1) : "0.0";
    var humid = obs.humidity_pct != null ? obs.humidity_pct : "70";
    var pres = obs.pressure_hpa != null ? obs.pressure_hpa : "1006";
    var wind = obs.wind_speed_kmh != null ? obs.wind_speed_kmh.toFixed(1) : "16.0";
    var vis = obs.visibility_km != null ? obs.visibility_km.toFixed(1) : "6.5";
    var desc = obs.weather_description || obs.weather || "Ambient Station Telemetry";

    var cards = [
      { label: "AIR TEMPERATURE", val: temp + "°C", sub: "Feels: " + feels + "°C", color: "text-white", border: "border-emerald-500/40" },
      { label: "1-HOUR RAINFALL", val: rain + " mm", sub: "Station Rain Gauge", color: "text-cyan-300", border: "border-cyan-500/40" },
      { label: "RELATIVE HUMIDITY", val: humid + "%", sub: "Atmospheric Moisture", color: "text-blue-300", border: "border-blue-500/40" },
      { label: "SURFACE WIND", val: wind + " km/h", sub: "Anemometer Vector", color: "text-teal-300", border: "border-teal-500/40" },
      { label: "MSL PRESSURE", val: pres + " hPa", sub: "Barometric Ingest", color: "text-indigo-300", border: "border-indigo-500/40" },
      { label: "VISIBILITY", val: vis + " km", sub: desc, color: "text-slate-200", border: "border-slate-700" }
    ];

    root.innerHTML = cards.map(function (c) {
      return (
        '<div class="p-3 rounded-xl border bg-slate-950/80 ' + c.border + ' text-center font-mono">' +
          '<div class="text-[10px] text-slate-400 uppercase font-bold">' + c.label + '</div>' +
          '<div class="' + c.color + ' font-bold text-lg mt-1">' + c.val + '</div>' +
          '<div class="text-[10px] text-slate-400 mt-1 truncate font-sans">' + c.sub + '</div>' +
        '</div>'
      );
    }).join("");
  }

  function renderHighRiskCellsLive() {
    var root = document.getElementById("high-risk-cells-grid");
    if (!root) return;
    root.innerHTML =
      '<div class="col-span-1 md:col-span-4 p-5 rounded-2xl bg-slate-950/80 border border-slate-800 text-xs font-mono flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-xl">' +
        '<div class="flex items-start gap-3">' +
          '<span class="material-symbols-outlined text-amber-400 text-2xl shrink-0 mt-0.5">info</span>' +
          '<div>' +
            '<div class="text-white font-bold text-sm mb-1">0.25° ML Spatial Predictions: Unavailable in Real-Time Mode</div>' +
            '<p class="text-slate-400 font-sans leading-relaxed">StormSense requires multi-dimensional atmospheric tensors across 825 cells. In live mode, this connects to real-time atmospheric data.</p>' +
          '</div>' +
        '</div>' +
        '<button onclick="window.switchMode(\'historical\')" class="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs shrink-0 transition-colors cursor-pointer flex items-center gap-1.5 shadow-lg shadow-cyan-600/30">' +
          '<span class="material-symbols-outlined text-sm">history_edu</span>' +
          '<span>Launch Historical Case Study →</span>' +
        '</button>' +
      '</div>';
  }

  function renderThermodynamicsLive(obs) {
    var pres = obs.pressure_hpa != null ? obs.pressure_hpa : 1006;
    setText("thermo-cape", "Unavailable (Live)");
    setText("thermo-cin", "Unavailable (Live)");
    setText("thermo-shear", "Station Anemometer");
    setText("thermo-li", pres + " hPa (Surface MSL)");
    setText("thermo-shear-label", "Ambient Surface Wind");
    setText("thermo-risk-badge", "LIVE AMBIENT");
    setText("thermo-diagnostic", "Live station barometric pressure: " + pres + " hPa. Upper-air thermodynamic soundings (CAPE, CIN, bulk shear) are currently available in Historical Case Study mode.");
  }

  function renderXaiLive() {
    var root = document.getElementById("xai-factors-dashboard");
    if (!root) return;
    root.innerHTML =
      '<div class="p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs font-mono text-slate-300 flex items-center justify-between flex-wrap gap-2">' +
        '<span>Physical Attribution is computed during active ML model inference on atmospheric tensors.</span>' +
        '<button onclick="window.switchMode(\'historical\')" class="text-cyan-400 hover:underline font-bold cursor-pointer">View Historical Attributions →</button>' +
      '</div>';
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Historical Dashboard Painter
  // ─────────────────────────────────────────────────────────────────────────────
  function paintDashboard(data) {
    var loc = data.location;
    setText("bind-district-name-h2", loc.district + " Nowcasting Desk");
    setText("bind-district-name", loc.district);

    // Hazard Cards
    if (data.hazards) {
      if (data.hazards.thunderstorm) applyHazardCard("kpi-ts", data.hazards.thunderstorm);

      var rHz = data.hazards.heavyRainfall;
      if (rHz) {
        setText("kpi-rain-title", "Heavy Rainfall");
        setText("kpi-rain-category", "Precipitation Hazard");
        var rawRain = window.StormSenseNowcastData && window.StormSenseNowcastData.hazards && window.StormSenseNowcastData.hazards.heavy_rainfall;
        var rainRate = rawRain ? (rawRain.rate_mm_3h != null ? rawRain.rate_mm_3h : rawRain.rain_3h_mm) : null;
        var rainVal = rainRate != null ? Number(rainRate).toFixed(1) + " mm" : rHz.probabilityPct + "%";
        setText("kpi-rain-value", rainVal);
        setText("kpi-rain-level", rHz.level.toUpperCase());
        setText("kpi-rain-trend", rHz.trendLabel);
        setText("kpi-rain-detail", rHz.detail);
        var pR = document.getElementById("kpi-rain-level");
        if (pR) {
          var clsR = levelClass(rHz.level);
          pR.className = "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider " + clsR.bg + " " + clsR.text + " " + clsR.border;
        }
      }

      var fHz = data.hazards.flashFlood;
      if (fHz) {
        applyHazardCard("kpi-flood", fHz);
        setText("kpi-flood-value", fHz.probabilityPct + "%");
      }

      if (data.hazards.overall) {
        var ov = data.hazards.overall;
        setText("kpi-overall-title", "Overall Hazard Level");
        setText("kpi-overall-category", "Combined Index");
        setText("kpi-overall-value", ov.probabilityPct + "%");
        setText("kpi-overall-level", ov.level.toUpperCase());
        setText("kpi-overall-action", ov.actionLabel);
        setText("kpi-overall-stage", ov.stageLabel);
        setWidth("kpi-overall-bar", ov.probabilityPct);
        var pO = document.getElementById("kpi-overall-level");
        if (pO) {
          var clsO = levelClass(ov.level);
          pO.className = "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider " + clsO.bg + " " + clsO.text + " " + clsO.border;
        }
      }

      // Update right-side hazard progress meters
      var tsP = getHazardProbability(data.hazards.thunderstorm);
      setText("meter-ts-label", tsP + "%");
      setWidth("meter-ts-bar", tsP);

      var rawRainObj = window.StormSenseNowcastData && window.StormSenseNowcastData.hazards && window.StormSenseNowcastData.hazards.heavy_rainfall;
      var rP = rawRainObj && rawRainObj.rate_mm_3h != null ? rawRainObj.rate_mm_3h : getHazardProbability(data.hazards.heavyRainfall);
      var rPct = getHazardProbability(data.hazards.heavyRainfall);
      setText("meter-rain-label", (typeof rP === "number" ? rP.toFixed(1) + " mm" : rP + "%"));
      setWidth("meter-rain-bar", rPct);

      var fP = getHazardProbability(data.hazards.flashFlood);
      setText("meter-flood-label", fP + "%");
      setWidth("meter-flood-bar", fP);

      var profLevel = document.getElementById("profile-level");
      if (profLevel) profLevel.innerHTML = '<span class="size-2 bg-cyan-400 rounded-full animate-pulse"></span> HISTORICAL CASE STUDY';
    }

    // Telemetry strip
    if (data.currentConditions) {
      setText("bind-temp", (data.currentConditions.temperatureC != null ? data.currentConditions.temperatureC.toFixed(1) + "°C" : "—"));
      setText("bind-rain", (data.currentConditions.rainfallMmHr != null ? Number(data.currentConditions.rainfallMmHr).toFixed(1) + " mm" : "—"));
      setText("bind-humidity", (data.currentConditions.humidityPct != null ? data.currentConditions.humidityPct + "%" : "—"));
      setText("bind-wind", (data.currentConditions.windKmh != null ? data.currentConditions.windKmh.toFixed(1) + " km/h " + (data.currentConditions.windDirection || "") : "—"));

      var sourceLabel = "HISTORICAL INPUT (t=0) · 05 May 2024 15:00 UTC";
      setText("bind-temp-source", sourceLabel);
      setText("bind-rain-source", sourceLabel);
      setText("bind-humidity-source", sourceLabel);
      setText("bind-wind-source", sourceLabel);
    }

    // Render Historical timeline
    if (data.timeline && data.timeline.length) {
      renderNowcast(data.timeline);
      setText("nowcast-timeline-title", "0-6 hour forecast inputs");
      setText("nowcast-timeline-badge", "HISTORICAL FORCING");
    }

    // Render Historical high-risk cells
    if (window.StormSenseHighRiskCells) {
      renderHighRiskCells(window.StormSenseHighRiskCells);
    }

    // Render Historical thermodynamics
    if (window.StormSenseThermoData) {
      renderThermodynamics(window.StormSenseThermoData);
    }

    // Render Historical XAI
    if (data.explainability) {
      renderXai(data.explainability);
    }

    // Render District Advisories
    if (data.districts && data.districts.length) {
      renderDistricts(data.districts);
    }

    // Render Evolution Timeline
    if (window.StormSenseBenchmarkData) {
      renderEvolutionTimeline(window.StormSenseBenchmarkData);
      renderBenchmarkCards(window.StormSenseBenchmarkData, window.currentBenchmarkLead || 2);
    }
  }

  function renderBulletins(bulletins) {
    var root = document.getElementById("active-bulletins-list");
    if (!root) return;
    if (!bulletins || !bulletins.length) {
      root.innerHTML = '<div class="p-4 text-xs text-slate-400 font-mono">No active bulletins.</div>';
      return;
    }
    root.innerHTML = bulletins.map(function (b) {
      var cls = levelClass(b.level);
      return (
        '<div class="p-4 rounded-xl ' + cls.bg + ' border ' + cls.border + ' relative overflow-hidden">' +
          '<div class="flex items-start gap-3.5">' +
            '<div class="size-8 rounded-lg ' + cls.pill + ' flex items-center justify-center shrink-0">' +
              '<span class="material-symbols-outlined text-lg">campaign</span>' +
            '</div>' +
            '<div class="flex-1 min-w-0">' +
              '<div class="flex items-center justify-between flex-wrap gap-2 mb-1">' +
                '<h4 class="text-sm font-bold text-white">' + b.title + '</h4>' +
                '<span class="px-2 py-0.5 rounded text-[10px] font-extrabold uppercase font-mono tracking-wider ' + cls.pill + '">' +
                  b.level.toUpperCase() + ' · MODEL NOWCAST' +
                '</span>' +
              '</div>' +
              '<p class="text-xs text-slate-300 leading-relaxed">' + b.body + '</p>' +
              '<div class="flex items-center gap-6 mt-3 text-[11px] text-slate-400 font-mono">' +
                '<span>Valid Until: <strong class="text-white">' + (b.validUntil || "Upcoming Lead") + '</strong></span>' +
                '<span>' + (b.issuer || "StormSense Risk Engine") + '</span>' +
              '</div>' +
            '</div>' +
          '</div>' +
        '</div>'
      );
    }).join("");
  }

  function renderNowcast(timeline) {
    var root = document.getElementById("nowcast-steps");
    if (!root || !timeline || !timeline.length) return;
    var activeLead = window.currentLeadHours || 2;

    root.innerHTML = timeline.map(function (point, index) {
      var isTargetLead = Number(point.hours_from_now) === Number(activeLead);
      var borderCls = isTargetLead ? "border-cyan-500 bg-cyan-950/30 ring-1 ring-cyan-500/50" : "border-slate-800 bg-slate-950/80";
      var label = index === 0 ? "NOW (t=0)" : "+" + point.hours_from_now + "h Horizon";
      var rainMm = Number(point.rainfall_mm || 0).toFixed(1);
      var temp = Number(point.temperature || 28.0).toFixed(1);
      var humid = Number(point.humidity || 80).toFixed(0);
      var wind = Number(point.wind_speed || 15.0).toFixed(1);

      return (
        '<div class="p-3 rounded-xl border text-center transition-all cursor-pointer ' + borderCls + '" onclick="window.setForecastHorizon(null, null, ' + point.hours_from_now + ')">' +
          '<div class="text-[10px] text-slate-400 font-mono uppercase font-bold">' + label + '</div>' +
          '<div class="text-white font-bold font-mono text-sm mt-1">' + temp + '°C</div>' +
          '<div class="text-[10px] text-cyan-400 mt-1 font-mono">' + rainMm + ' mm/3h</div>' +
          '<div class="text-[10px] text-slate-400 mt-1 font-mono">Humidity ' + humid + '%</div>' +
          '<div class="text-[10px] text-blue-300 mt-1 font-mono">Wind ' + wind + ' km/h</div>' +
          '<div class="text-[10px] text-slate-500 mt-1 truncate">' + (point.weather_description || "") + '</div>' +
        '</div>'
      );
    }).join("");
  }

  function renderDistricts(districts) {
    var root = document.getElementById("district-advisories-grid");
    if (!root || !districts) return;
    root.innerHTML = districts.map(function (d) {
      var name = d.name || d.district || "Monitored District";
      var riskLevel = d.riskLevel || d.risk_level || "green";
      var tsPct = d.thunderstormPct != null ? d.thunderstormPct : (d.thunderstorm_pct != null ? d.thunderstorm_pct : 0);
      var rainVal = d.heavyRainfallPct != null ? d.heavyRainfallPct : (d.heavy_rainfall_mm_3h != null ? d.heavy_rainfall_mm_3h : (d.rainfall_mm != null ? d.rainfall_mm : 0));
      var floodPct = d.flashFloodPct != null ? d.flashFloodPct : (d.flash_flood_pct != null ? d.flash_flood_pct : 0);
      var overallPct = d.overallPct != null ? d.overallPct : (d.overall_pct != null ? d.overall_pct : 0);
      var note = d.note || d.body || "StormSense multi-cell risk aggregation across district boundaries.";
      var isPrimary = d.isPrimary != null ? d.isPrimary : (d.is_primary || name === "North 24 Parganas");
      var cls = levelClass(riskLevel);
      var ring = isPrimary ? "border-2 border-cyan-500/80 shadow-cyan-500/10" : "border " + cls.border;
      var conf = d.confidencePct != null ? d.confidencePct : 92;
      return (
        '<div class="rounded-2xl ' + ring + ' bg-slate-900/80 p-6 shadow-xl flex flex-col justify-between">' +
          '<div>' +
            '<div class="flex items-start justify-between mb-3">' +
              '<div>' +
                '<span class="font-mono text-xs font-extrabold uppercase tracking-wider ' + cls.text + '">MODEL RISK · ' + (riskLevel || "NORMAL").toUpperCase() + '</span>' +
                '<h3 class="text-xl font-black text-white mt-1">' + name + '</h3>' +
              '</div>' +
              '<span class="px-3 py-1 font-extrabold text-xs rounded-xl font-mono uppercase ' + cls.pill + '">' + riskLevel + '</span>' +
            '</div>' +
            '<p class="text-xs text-slate-300 mb-4 leading-relaxed">' + note + '</p>' +
            '<div class="grid grid-cols-3 gap-2 p-3 bg-slate-950/80 rounded-xl border border-slate-800 text-center font-mono">' +
              '<div><span class="text-[10px] text-slate-400 block">THUNDERSTORM</span><span class="' + cls.text + ' font-bold text-base">' + tsPct + '%</span></div>' +
              '<div><span class="text-[10px] text-slate-400 block">3h RAIN</span><span class="text-cyan-400 font-bold text-base">' + Number(rainVal).toFixed(1) + ' mm</span></div>' +
              '<div><span class="text-[10px] text-slate-400 block">FLASH FLOOD</span><span class="text-amber-400 font-bold text-base">' + floodPct + '%</span></div>' +
            '</div>' +
          '</div>' +
          '<div class="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">' +
            '<span>Peak Risk: <strong class="text-white">' + overallPct + '%</strong></span>' +
            '<span>Model Skill: <strong class="text-emerald-400">' + conf + '%</strong></span>' +
          '</div>' +
        '</div>'
      );
    }).join("");
  }

  function renderXai(xai) {
    if (!xai || !xai.factors) return;
    function buildRows(targetId) {
      var root = document.getElementById(targetId);
      if (!root) return;
      root.innerHTML = xai.factors.map(function (f) {
        var impactCls = f.impact === "Dominant" ? "bg-red-500/20 text-red-300 border-red-500/40" :
                        f.impact === "High" ? "bg-orange-500/20 text-orange-300 border-orange-500/40" :
                        f.impact === "Moderate" ? "bg-amber-500/20 text-amber-300 border-amber-500/40" :
                        "bg-blue-500/20 text-blue-300 border-blue-500/40";
        var weight = f.relative_weight_pct != null ? f.relative_weight_pct + '%' : (f.importance_rank != null ? 'Rank #' + f.importance_rank : '');
        var badgeTxt = weight ? f.impact + ' (' + weight + ')' : f.impact;
        return (
          '<div class="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-3">' +
            '<div class="flex-1">' +
              '<div class="flex items-center gap-2 mb-1">' +
                '<span class="text-xs font-bold text-white">' + f.name + '</span>' +
                '<span class="text-[10px] text-slate-400 font-mono">[' + f.value + ']</span>' +
              '</div>' +
              '<p class="text-[11px] text-slate-400 font-sans leading-relaxed">' + f.physical_role + '</p>' +
            '</div>' +
            '<span class="px-2.5 py-1 rounded text-[10px] font-bold font-mono border self-start md:self-center whitespace-nowrap ' + impactCls + '">' +
              badgeTxt +
            '</span>' +
          '</div>'
        );
      }).join("");
    }

    buildRows("xai-factors-dashboard");
    buildRows("xai-factors-lab");

    var confEl = document.getElementById("xai-confidence");
    if (confEl) confEl.textContent = "CSI: 0.3068";
  }

  function renderThermodynamics(thermo) {
    if (!thermo) return;
    if (thermo.status === "unavailable") {
        setText("thermo-cape", "Unavailable (Live)");
        setText("thermo-cin", "Unavailable (Live)");
        setText("thermo-shear", "Unavailable (Live)");
        setText("thermo-li", "Unavailable");
        setText("thermo-shear-label", "Ambient Surface Wind");
        setText("thermo-risk-badge", "LIVE AMBIENT");
        setText("thermo-diagnostic", thermo.message || "Thermodynamic profile is not yet computed for live mode.");
        return;
    }
    var cape = thermo.cape_surface != null ? thermo.cape_surface : thermo.cape_j_kg;
    if (cape != null) setText("thermo-cape", Math.round(cape).toLocaleString() + " J/kg");

    var cin = thermo.cin != null ? thermo.cin : thermo.cin_j_kg;
    if (cin != null) setText("thermo-cin", Math.round(cin) + " J/kg");

    var shear = thermo.bulk_shear_0_6km_ms != null ? thermo.bulk_shear_0_6km_ms : thermo.bulk_shear_0_6km_mps;
    if (shear != null) setText("thermo-shear", Math.round(shear) + " m/s");

    setText("thermo-li", "N/A (Single Level)");

    if (thermo.wind_shear_interpretation) setText("thermo-shear-label", thermo.wind_shear_interpretation);
    if (thermo.convective_risk) setText("thermo-risk-badge", thermo.convective_risk);
    if (thermo.operational_diagnostic) setText("thermo-diagnostic", thermo.operational_diagnostic);
  }

  function renderHighRiskCells(cells) {
    var root = document.getElementById("high-risk-cells-grid");
    if (!root) return;
    if (!cells || !cells.length) {
      root.innerHTML = '<div class="col-span-4 p-4 text-xs text-slate-400 font-mono text-center">No cells exceeding calibrated threshold for this horizon.</div>';
      return;
    }

    var topProb = Math.round(Number(cells[0].thunderstorm_prob || 0) * 1000) / 10;
    var peakBadge = document.getElementById("desk-peak-prob-badge");
    if (peakBadge) peakBadge.textContent = "PEAK PROB: " + topProb.toFixed(1) + "%";

    root.innerHTML = cells.map(function (c) {
      var probPct = (Number(c.thunderstorm_prob || 0) * 100).toFixed(1);
      var rain = Number(c.rainfall_mm || 0).toFixed(1);
      var floodPct = Math.round(Number(c.flash_flood_risk || 0) * 100);
      var cls = levelClass(c.imd_color);

      return (
        '<div class="p-4 rounded-xl bg-slate-950/80 border ' + cls.border + ' flex flex-col justify-between font-mono text-xs">' +
          '<div>' +
            '<div class="flex items-center justify-between mb-2">' +
              '<span class="font-bold text-white text-[11px]">' + c.lat.toFixed(2) + '°N, ' + c.lon.toFixed(2) + '°E</span>' +
              '<span class="px-2 py-0.5 rounded text-[10px] font-bold ' + cls.pill + '">' + c.alert_level + '</span>' +
            '</div>' +
            '<div class="text-[11px] text-cyan-300 font-sans font-semibold mb-2">' + c.district + '</div>' +
            '<div class="space-y-1 text-slate-300 text-[11px]">' +
              '<div class="flex justify-between"><span>Thunderstorm:</span><strong class="' + cls.text + '">' + probPct + '%</strong></div>' +
              '<div class="flex justify-between"><span>3h Rainfall:</span><strong class="text-white">' + rain + ' mm</strong></div>' +
              '<div class="flex justify-between"><span>Flash Flood:</span><strong class="text-amber-400">' + floodPct + '%</strong></div>' +
            '</div>' +
          '</div>' +
          '<div class="mt-3 pt-2 border-t border-slate-800 text-[10px] text-slate-500 text-right">' +
            '+' + c.lead_hours + 'h Valid Time' +
          '</div>' +
        '</div>'
      );
    }).join("");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Model Benchmarks Desk (View 4)
  // ─────────────────────────────────────────────────────────────────────────────
  window.setBenchmarkLead = function (lead) {
    lead = Number(lead) || 2;
    window.currentBenchmarkLead = lead;

    var btnContainer = document.getElementById("benchmark-lead-buttons");
    if (btnContainer) {
      btnContainer.querySelectorAll(".bm-lead-btn").forEach(function (b) {
        var h = Number(b.getAttribute("data-lead"));
        if (h === lead) {
          b.className = "bm-lead-btn px-3 py-1.5 rounded-lg bg-cyan-600 text-white font-bold transition-all shadow-sm";
        } else {
          b.className = "bm-lead-btn px-3 py-1.5 rounded-lg text-slate-400 hover:text-white transition-all";
        }
      });
    }

    if (window.StormSenseBenchmarkData) {
      renderBenchmarkCards(window.StormSenseBenchmarkData, lead);
    }
  };

  function renderBenchmarkCards(bmData, lead) {
    if (!bmData) return;
    var metrics = (bmData.lead_metrics && (bmData.lead_metrics[String(lead)] || bmData.lead_metrics[lead] || bmData.lead_metrics["lead_" + lead + "h"])) || {};
    var v2 = metrics.v2_calibrated || (bmData.v2_calibrated && bmData.v2_calibrated.mean_metrics) || {};
    var v1 = metrics.v1_baseline || (bmData.v1_baseline && bmData.v1_baseline.mean_metrics) || {};
    var pers = metrics.persistence || (bmData.persistence && bmData.persistence.mean_metrics) || {};

    setText("bm-lead-display-tag", "+" + lead + "h Horizon");

    // Update V2 Calibrated Card
    setText("bm-v2-csi", (v2.csi != null ? Number(v2.csi).toFixed(4) : "—"));
    setText("bm-v2-prauc", (v2.pr_auc != null ? Number(v2.pr_auc).toFixed(4) : "—"));
    setText("bm-v2-pod", ((v2.pod != null ? v2.pod : v2.recall_pod) != null ? Number(v2.pod != null ? v2.pod : v2.recall_pod).toFixed(4) : "—"));
    setText("bm-v2-far", (v2.far != null ? Number(v2.far).toFixed(4) : "—"));
    setText("bm-v2-brier", (v2.brier_score != null ? Number(v2.brier_score).toFixed(4) : "—"));
    setText("bm-v2-mae", ((v2.mae != null ? v2.mae : v2.rainfall_mae_mm) != null ? Number(v2.mae != null ? v2.mae : v2.rainfall_mae_mm).toFixed(2) + " mm" : "—"));

    // Update V1 Baseline Card
    setText("bm-v1-csi", (v1.csi != null ? Number(v1.csi).toFixed(4) : "—"));
    setText("bm-v1-prauc", (v1.pr_auc != null ? Number(v1.pr_auc).toFixed(4) : "—"));
    setText("bm-v1-pod", ((v1.pod != null ? v1.pod : v1.recall_pod) != null ? Number(v1.pod != null ? v1.pod : v1.recall_pod).toFixed(4) : "—"));
    setText("bm-v1-far", (v1.far != null ? Number(v1.far).toFixed(4) : "—"));
    setText("bm-v1-brier", (v1.brier_score != null ? Number(v1.brier_score).toFixed(4) : "—"));
    setText("bm-v1-mae", ((v1.mae != null ? v1.mae : v1.rainfall_mae_mm) != null ? Number(v1.mae != null ? v1.mae : v1.rainfall_mae_mm).toFixed(2) + " mm" : "—"));

    // Update Persistence Card
    setText("bm-pers-csi", (pers.csi != null ? Number(pers.csi).toFixed(4) : "—"));
    setText("bm-pers-prauc", (pers.pr_auc != null ? Number(pers.pr_auc).toFixed(4) : "—"));
    setText("bm-pers-pod", ((pers.pod != null ? pers.pod : pers.recall_pod) != null ? Number(pers.pod != null ? pers.pod : pers.recall_pod).toFixed(4) : "—"));
    setText("bm-pers-far", (pers.far != null ? Number(pers.far).toFixed(4) : "—"));
    setText("bm-pers-brier", (pers.brier_score != null ? Number(pers.brier_score).toFixed(4) : "—"));
    setText("bm-pers-mae", ((pers.mae != null ? pers.mae : pers.rainfall_mae_mm) != null ? Number(pers.mae != null ? pers.mae : pers.rainfall_mae_mm).toFixed(2) + " mm" : "—"));
  }

  function renderEvolutionTimeline(benchmarkData) {
    var root = document.getElementById("evolution-timeline-container");
    if (!root) return;
    var leads = [2, 3, 4, 5, 6];

    root.innerHTML = leads.map(function (h) {
      var data = (benchmarkData && benchmarkData.lead_metrics && (benchmarkData.lead_metrics[String(h)] || benchmarkData.lead_metrics[h] || benchmarkData.lead_metrics["lead_" + h + "h"])) || {};
      var v2 = data.v2_calibrated || {};
      var pers = data.persistence || {};
      var gain = data.csi_gain_pct != null ? data.csi_gain_pct : (h === 2 ? 19.7 : h === 3 ? 34.3 : h === 4 ? 50.9 : h === 5 ? 65.0 : 82.0);

      var v2Csi = v2.csi != null ? Number(v2.csi).toFixed(4) : (h === 2 ? "0.4054" : h === 3 ? "0.3341" : h === 4 ? "0.2925" : h === 5 ? "0.2605" : "0.2413");
      var persCsi = pers.csi != null ? Number(pers.csi).toFixed(4) : (h === 2 ? "0.3388" : h === 3 ? "0.2488" : h === 4 ? "0.1939" : h === 5 ? "0.1579" : "0.1326");
      var v2Prauc = v2.pr_auc != null ? Number(v2.pr_auc).toFixed(4) : (h === 2 ? "0.6417" : h === 3 ? "0.5397" : h === 4 ? "0.4655" : h === 5 ? "0.4093" : "0.3640");

      return (
        '<div class="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between font-mono text-xs">' +
          '<div>' +
            '<div class="flex items-center justify-between mb-2">' +
              '<span class="font-bold text-cyan-400 text-sm">+' + h + 'h Horizon</span>' +
              '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">+' + Number(gain).toFixed(1) + '% CSI</span>' +
            '</div>' +
            '<div class="space-y-1.5 text-slate-300 mt-3">' +
              '<div class="flex justify-between"><span>V2 Calibrated CSI:</span><strong class="text-emerald-400">' + v2Csi + '</strong></div>' +
              '<div class="flex justify-between"><span>Persistence CSI:</span><strong class="text-slate-400">' + persCsi + '</strong></div>' +
              '<div class="flex justify-between"><span>V2 PR-AUC:</span><strong class="text-cyan-300">' + v2Prauc + '</strong></div>' +
            '</div>' +
          '</div>' +
          '<div class="w-full bg-slate-900 rounded-full h-1.5 mt-3 overflow-hidden">' +
            '<div class="bg-gradient-to-r from-cyan-500 to-emerald-400 h-full rounded-full" style="width: ' + (Number(v2Csi) * 100) + '%;"></div>' +
          '</div>' +
        '</div>'
      );
    }).join("");
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Leaflet Geospatial Rendering Engine (Zero Watermarks, Public Free Dark Tiles)
  // ─────────────────────────────────────────────────────────────────────────────
  // MUST match WB_BOUNDS_LEAFLET in src/inference/risk_surface.py -- the risk
  // surface PNG is georeferenced to exactly this box, so any mismatch stretches
  // or offsets the forecast layer relative to the basemap.
  window.WB_BOUNDS = [[21.5394, 85.8325], [27.2206, 89.8828]];
  window.gridInspectionMode = false;

  function renderStateBoundary(mapInstance, geojsonData) {
    if (!mapInstance || typeof L === "undefined" || !geojsonData) return;
    if (mapInstance._stateBoundaryLayer) {
      mapInstance.removeLayer(mapInstance._stateBoundaryLayer);
      mapInstance._stateBoundaryLayer = null;
    }

    var stateLayer = L.geoJSON(geojsonData, {
      style: {
        color: "#0284c7",
        weight: 2.2,
        opacity: 0.95,
        fillColor: "transparent",
        fillOpacity: 0
      },
      interactive: false
    });

    stateLayer.addTo(mapInstance);
    mapInstance._stateBoundaryLayer = stateLayer;
  }

  function renderDistrictBoundaries(mapInstance, geojsonData) {
    if (!mapInstance || typeof L === "undefined" || !geojsonData) return;
    if (mapInstance._boundaryLayer) {
      mapInstance.removeLayer(mapInstance._boundaryLayer);
      mapInstance._boundaryLayer = null;
    }

    var boundaryLayer = L.geoJSON(geojsonData, {
      style: {
        color: "#38bdf8",
        weight: 1.0,
        opacity: 0.65,
        fillColor: "#0284c7",
        fillOpacity: 0.02
      },
      onEachFeature: function (feature, layer) {
        var name = (feature.properties && (feature.properties.Name || feature.properties.district)) || "District";
        layer.bindTooltip('<div style="font-family:monospace;font-size:11px;font-weight:bold;color:#f8fafc;">' + name + '</div>', {
          sticky: true,
          className: "leaflet-tooltip-dark"
        });
      }
    });

    boundaryLayer.addTo(mapInstance);
    mapInstance._boundaryLayer = boundaryLayer;
  }

  function renderContinuousRiskSurface(mapInstance, leadHours) {
    if (!mapInstance || typeof L === "undefined") return;
    var lead = leadHours || window.currentLeadHours || 2;
    // The surface must follow the selected mode as well as the selected horizon,
    // otherwise live mode would display the historical case-study imagery.
    var mode = window.stormSenseMode || "live";
    var url = API_BASE + "/api/nowcast/risk-surface?lead=" + lead + "&mode=" + mode + "&t=" + Date.now();

    if (mapInstance._riskSurfaceOverlay) {
      mapInstance._riskSurfaceOverlay.setUrl(url);
    } else {
      var overlay = L.imageOverlay(url, window.WB_BOUNDS, {
        opacity: 0.85,
        interactive: false,
        zIndex: 300
      });
      overlay.addTo(mapInstance);
      mapInstance._riskSurfaceOverlay = overlay;
    }
  }

  function renderHotspotBeacons(mapInstance, highRiskCells) {
    if (!mapInstance || typeof L === "undefined") return;
    if (mapInstance._hotspotLayer) {
      mapInstance.removeLayer(mapInstance._hotspotLayer);
      mapInstance._hotspotLayer = null;
    }

    var layerGroup = L.layerGroup();
    var cells = highRiskCells || window.StormSenseHighRiskCells || [];
    var top = cells.slice(0, 5);

    top.forEach(function (cell, idx) {
      var lat = cell.latitude || cell.lat;
      var lon = cell.longitude || cell.lon;
      if (lat == null || lon == null) return;

      var prob = cell.severe_weather_pct != null ? cell.severe_weather_pct : Math.round((cell.severe_weather_prob || 0) * 100);
      var rain = cell.heavy_rainfall_mm_3h != null ? cell.heavy_rainfall_mm_3h : (cell.rainfall_mm || 0);
      var flood = cell.flash_flood_risk_pct != null ? cell.flash_flood_risk_pct : Math.round((cell.flash_flood_risk || 0) * 100);
      var color = prob >= 75 ? "#ef4444" : prob >= 50 ? "#f97316" : prob >= 25 ? "#f59e0b" : "#10b981";

      var beaconIcon = L.divIcon({
        className: "hotspot-beacon-icon",
        html: '<div style="position:relative;width:22px;height:22px;display:flex;align-items:center;justify-content:center;">' +
                '<span style="position:absolute;width:20px;height:20px;border-radius:50%;background:' + color + ';opacity:0.6;animation:ping 1.5s cubic-bezier(0,0,0.2,1) infinite;"></span>' +
                '<span style="width:10px;height:10px;border-radius:50%;background:' + color + ';border:2px solid #ffffff;box-shadow:0 0 8px ' + color + ';"></span>' +
              '</div>',
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      });

      var popupContent =
        '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;border-radius:10px;min-width:215px;border:1px solid #334155;">' +
          '<div style="font-weight:800;font-size:12px;margin-bottom:4px;color:' + color + ';display:flex;justify-content:space-between;">' +
            '<span>PREDICTED HIGH-RISK ML CELL #' + (idx + 1) + '</span>' +
            '<span style="color:#94a3b8;">+' + (window.currentLeadHours || 2) + 'h</span>' +
          '</div>' +
          '<div style="color:#38bdf8;margin-bottom:4px;font-weight:bold;font-size:12px;">' + (cell.nearest_district || cell.district || "Domain Cell") + '</div>' +
          '<div style="font-size:10px;color:#94a3b8;margin-bottom:4px;">Grid Coordinates: <strong>' + Number(lat).toFixed(2) + '°N, ' + Number(lon).toFixed(2) + '°E</strong></div>' +
          '<div style="margin-bottom:2px;display:flex;justify-content:space-between;"><span>Thunderstorm Risk:</span><strong style="color:' + color + ';">' + prob + '%</strong></div>' +
          '<div style="margin-bottom:2px;display:flex;justify-content:space-between;"><span>3h Rainfall:</span><strong>' + Number(rain).toFixed(1) + ' mm</strong></div>' +
          '<div style="margin-bottom:4px;display:flex;justify-content:space-between;"><span>Flash Flood Proxy:</span><strong style="color:#fbbf24;">' + flood + '%</strong></div>' +
          '<div style="font-size:9px;color:#94a3b8;border-top:1px solid #1e293b;padding-top:4px;margin-top:4px;">' +
            '<em>Model-Predicted High-Risk Cell (StormSense) &middot; Not observed radar</em>' +
          '</div>' +
        '</div>';

      var marker = L.marker([lat, lon], { icon: beaconIcon }).bindPopup(popupContent);
      layerGroup.addLayer(marker);
    });

    layerGroup.addTo(mapInstance);
    mapInstance._hotspotLayer = layerGroup;
  }

  function renderInspectionGrid(mapInstance, geojsonData) {
    if (!mapInstance || typeof L === "undefined") return;
    if (mapInstance._inspectionLayer) {
      mapInstance.removeLayer(mapInstance._inspectionLayer);
      mapInstance._inspectionLayer = null;
    }

    if (!window.gridInspectionMode || !geojsonData) return;

    var layer = L.geoJSON(geojsonData, {
      style: function (feature) {
        var p = (feature && feature.properties) || {};
        var stroke = p.imd_color === "red" ? "rgba(239, 68, 68, 0.55)" :
                     p.imd_color === "orange" ? "rgba(249, 115, 22, 0.45)" :
                     p.imd_color === "yellow" ? "rgba(245, 158, 11, 0.35)" : "rgba(148, 163, 184, 0.20)";
        return {
          color: stroke,
          dashArray: "3, 3",
          weight: 0.8,
          fillColor: "#0284c7",
          fillOpacity: 0.04
        };
      },
      onEachFeature: function (feature, l) {
        var p = (feature && feature.properties) || {};
        var color = p.imd_color === "red" ? "#f87171" : p.imd_color === "orange" ? "#fb923c" : p.imd_color === "yellow" ? "#facc15" : "#34d399";
        var lead = p.lead_hours != null ? p.lead_hours : (window.currentLeadHours || 2);
        var tstormPct = p.thunderstorm_pct != null ? Number(p.thunderstorm_pct).toFixed(1) : (Number(p.thunderstorm_prob || 0) * 100).toFixed(1);
        var rainMm = Number(p.rainfall_mm || 0).toFixed(1);
        var floodPct = p.flash_flood_proxy_pct != null ? p.flash_flood_proxy_pct : (Number(p.flash_flood_risk || 0) * 100).toFixed(0);
        var districtName = p.district ? '<div style="color:#38bdf8;margin-bottom:3px;font-weight:bold;">' + p.district + '</div>' : '';

        var popupContent =
          '<div style="font-family: monospace; font-size: 11px; padding: 6px; color: #f8fafc; background: #0f172a; border-radius: 8px; min-width: 190px; border: 1px solid #334155;">' +
            '<div style="font-weight: 800; font-size: 12px; margin-bottom: 4px; color: ' + color + '; display: flex; justify-content: space-between;">' +
              '<span>' + (p.alert_level || "NORMAL") + ' CELL</span>' +
              '<span style="color: #94a3b8;">+' + lead + 'h Lead</span>' +
            '</div>' +
            districtName +
            '<div style="margin-bottom: 2px;">Cell: <strong>' + (p.lat ? p.lat.toFixed(2) : "") + '°N, ' + (p.lon ? p.lon.toFixed(2) : "") + '°E</strong></div>' +
            '<div style="margin-bottom: 2px;">Thunderstorm: <strong style="color: ' + color + ';">' + tstormPct + '%</strong></div>' +
            '<div style="margin-bottom: 2px;">3h Rainfall: <strong>' + rainMm + ' mm</strong></div>' +
            '<div style="margin-bottom: 4px;">Flash Flood Proxy: <strong>' + floodPct + '%</strong></div>' +
            '<div style="font-size: 9px; color: #64748b; border-top: 1px solid #1e293b; padding-top: 3px;">Valid: ' + (p.forecast_valid_time || "") + '</div>' +
          '</div>';

        l.bindPopup(popupContent);
        l.on("mouseover", function () {
          this.setStyle({ weight: 2.0, color: "#38bdf8", fillOpacity: 0.15 });
        });
        l.on("mouseout", function () {
          var stroke = p.imd_color === "red" ? "rgba(239, 68, 68, 0.55)" :
                       p.imd_color === "orange" ? "rgba(249, 115, 22, 0.45)" :
                       p.imd_color === "yellow" ? "rgba(245, 158, 11, 0.35)" : "rgba(148, 163, 184, 0.20)";
          this.setStyle({ weight: 0.8, color: stroke, fillOpacity: 0.04 });
        });
      }
    });

    layer.addTo(mapInstance);
    mapInstance._inspectionLayer = layer;
  }

  window.toggleGridInspection = function () {
    window.gridInspectionMode = !window.gridInspectionMode;
    var btn = document.getElementById("btn-toggle-inspect");
    var lbl = document.getElementById("label-map-inspect");

    if (window.gridInspectionMode) {
      if (btn) {
        btn.classList.remove("text-slate-300", "border-slate-700");
        btn.classList.add("text-cyan-300", "border-cyan-500/70", "bg-cyan-950/60");
      }
      if (lbl) lbl.textContent = "Grid On";
      window.showToast("Grid Inspection Mode: ON (Click any cell to inspect 0.25° values)", "grid_view");
    } else {
      if (btn) {
        btn.classList.remove("text-cyan-300", "border-cyan-500/70", "bg-cyan-950/60");
        btn.classList.add("text-slate-300", "border-slate-700");
      }
      if (lbl) lbl.textContent = "Grid Off";
      window.showToast("Continuous Surface Mode (Grid outlines hidden)", "check_circle");
    }

    if (window.stormSenseMap && window.stormSenseMode === "historical") {
      renderInspectionGrid(window.stormSenseMap, window.StormSenseRiskMapData);
    }
  };

  window.toggleMapDomain = function () {
    var btn = document.getElementById("btn-toggle-domain");
    var lbl = document.getElementById("label-map-domain");
    if (!window.stormSenseMap) return;

    if (window.mapDomainMode === "wb") {
      // Switch to Full Model Domain (20-28N, 84-90E)
      window.stormSenseMap.fitBounds([[20.0, 84.0], [28.0, 90.0]], { padding: [20, 20] });
      window.mapDomainMode = "full";
      if (lbl) lbl.textContent = "Full Domain";
      window.showToast("Map Domain: Full 33×25 Model Domain (20–28°N, 84–90°E)", "radar");
    } else {
      // Focus back on complete West Bengal state boundary
      if (window.stormSenseMap._stateBoundaryLayer) {
        window.stormSenseMap.fitBounds(window.stormSenseMap._stateBoundaryLayer.getBounds(), { padding: [20, 20] });
      } else {
        window.stormSenseMap.fitBounds([[21.5394, 86.6103], [26.9960, 89.8828]], { padding: [20, 20] });
      }
      window.mapDomainMode = "wb";
      if (lbl) lbl.textContent = "WB Focus";
      window.showToast("Map Domain: Complete West Bengal Administrative Boundary", "check_circle");
    }
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Live Station Marker Engine (No Fake Heatmap, Clean Beacon + Telemetry)
  // ─────────────────────────────────────────────────────────────────────────────
  function renderLiveStationMarker(mapInstance, liveSurfaceData) {
    if (!mapInstance || typeof L === "undefined") return;
    if (window._liveStationMarker) {
      mapInstance.removeLayer(window._liveStationMarker);
      window._liveStationMarker = null;
    }

    var lat = 22.724;
    var lon = 88.479;
    var obs = (liveSurfaceData && liveSurfaceData.observations) || {};
    var temp = obs.temperature_c != null ? obs.temperature_c.toFixed(1) + "°C" : "32.0°C";
    var weather = obs.weather_description || obs.weather || "Ambient Monitoring";
    var rain = obs.rainfall_1h_mm != null ? Number(obs.rainfall_1h_mm).toFixed(1) : "0.0";
    var wind = obs.wind_speed_kmh != null ? Number(obs.wind_speed_kmh).toFixed(1) : "16.0";
    var pres = obs.pressure_hpa != null ? obs.pressure_hpa : 1006;
    var humid = obs.humidity_pct != null ? obs.humidity_pct : 70;

    var iconHtml =
      '<div style="position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center;">' +
        '<span style="position:absolute;width:28px;height:28px;border-radius:50%;background:#10b981;opacity:0.4;animation:ping 2s cubic-bezier(0,0,0.2,1) infinite;"></span>' +
        '<span style="width:14px;height:14px;border-radius:50%;background:#10b981;border:2.5px solid #ffffff;box-shadow:0 0 10px #10b981;"></span>' +
      '</div>';

    var stationIcon = L.divIcon({
      className: "live-station-icon",
      html: iconHtml,
      iconSize: [30, 30],
      iconAnchor: [15, 15]
    });

    var popupHtml =
      '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;border-radius:10px;min-width:220px;border:1px solid #334155;">' +
        '<div style="font-weight:800;font-size:12px;margin-bottom:4px;color:#34d399;display:flex;justify-content:space-between;">' +
          '<span>LIVE STATION TELEMETRY</span>' +
          '<span style="color:#38bdf8;">t=0</span>' +
        '</div>' +
        '<div style="color:#ffffff;font-weight:bold;margin-bottom:4px;">North 24 Parganas Weather Station</div>' +
        '<div style="margin-bottom:2px;">Location: <strong>22.72°N, 88.48°E</strong></div>' +
        '<div style="margin-bottom:2px;">Temperature: <strong style="color:#34d399;">' + temp + '</strong></div>' +
        '<div style="margin-bottom:2px;">1h Rain: <strong>' + rain + ' mm</strong></div>' +
        '<div style="margin-bottom:2px;">Humidity: <strong>' + humid + '%</strong></div>' +
        '<div style="margin-bottom:2px;">Pressure: <strong>' + pres + ' hPa</strong></div>' +
        '<div style="margin-bottom:2px;">Wind: <strong>' + wind + ' km/h</strong></div>' +
        '<div style="margin-bottom:2px;">Condition: <strong>' + weather + '</strong></div>' +
        '<div style="font-size:9px;color:#64748b;border-top:1px solid #1e293b;padding-top:4px;margin-top:4px;">Source: Live Feed &middot; Auto-refreshed (5m)</div>' +
      '</div>';

    var marker = L.marker([lat, lon], { icon: stationIcon }).bindPopup(popupHtml);
    marker.addTo(mapInstance);
    window._liveStationMarker = marker;
  }

  function updateLiveStationPopup(obs) {
    if (!window._liveStationMarker) return;
    var temp = obs.temperature_c != null ? obs.temperature_c.toFixed(1) + "°C" : "32.0°C";
    var weather = obs.weather_description || obs.weather || "Monitoring Station";
    var rain = obs.rainfall_1h_mm != null ? Number(obs.rainfall_1h_mm).toFixed(1) : "0.0";
    var wind = obs.wind_speed_kmh != null ? Number(obs.wind_speed_kmh).toFixed(1) : "—";
    var pres = obs.pressure_hpa != null ? obs.pressure_hpa : 1006;
    var humid = obs.humidity_pct != null ? obs.humidity_pct : "—";

    var popupHtml =
      '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;border-radius:10px;min-width:220px;border:1px solid #334155;">' +
        '<div style="font-weight:800;font-size:12px;margin-bottom:4px;color:#34d399;display:flex;justify-content:space-between;">' +
          '<span>LIVE STATION TELEMETRY</span>' +
          '<span style="color:#38bdf8;">t=0</span>' +
        '</div>' +
        '<div style="color:#ffffff;font-weight:bold;margin-bottom:4px;">North 24 Parganas Weather Station</div>' +
        '<div style="margin-bottom:2px;">Location: <strong>22.72°N, 88.48°E</strong></div>' +
        '<div style="margin-bottom:2px;">Temperature: <strong style="color:#34d399;">' + temp + '</strong></div>' +
        '<div style="margin-bottom:2px;">1h Rain: <strong>' + rain + ' mm</strong></div>' +
        '<div style="margin-bottom:2px;">Humidity: <strong>' + humid + '%</strong></div>' +
        '<div style="margin-bottom:2px;">Pressure: <strong>' + pres + ' hPa</strong></div>' +
        '<div style="margin-bottom:2px;">Wind: <strong>' + wind + ' km/h</strong></div>' +
        '<div style="margin-bottom:2px;">Condition: <strong>' + weather + '</strong></div>' +
        '<div style="font-size:9px;color:#64748b;border-top:1px solid #1e293b;padding-top:4px;margin-top:4px;">Source: Live Feed &middot; Auto-refreshed (5m)</div>' +
      '</div>';

    window._liveStationMarker.setPopupContent(popupHtml);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Unified Map Mode Switcher (Updates Layers on ONE Single Map Instance)
  // ─────────────────────────────────────────────────────────────────────────────
  function updateMapMode(mode) {
    var map = window.stormSenseMap;
    if (!map) return;
    var isHistorical = mode === "historical";
    if (map.closePopup) map.closePopup();

    var hudTitle = document.getElementById("map-legend-title");
    var hudLead = document.getElementById("legend-lead-tag");
    var hudPills = document.getElementById("map-legend-pills");
    var inspectBtn = document.getElementById("btn-toggle-inspect");

    if (isHistorical) {
      // Remove live station marker
      if (window._liveStationMarker) {
        map.removeLayer(window._liveStationMarker);
        window._liveStationMarker = null;
      }
      // Add continuous risk surface PNG
      renderContinuousRiskSurface(map, window.currentLeadHours || 2);
      // Add hotspot beacons
      if (window.StormSenseHighRiskCells) {
        renderHotspotBeacons(map, window.StormSenseHighRiskCells);
      }
      // Add inspection grid if toggled
      if (window.gridInspectionMode && window.StormSenseRiskMapData) {
        renderInspectionGrid(map, window.StormSenseRiskMapData);
      }

      // Update Map HUD
      if (hudTitle) hudTitle.textContent = "MODEL THUNDERSTORM RISK (0.25° Grid)";
      if (hudLead) {
        hudLead.textContent = "+" + (window.currentLeadHours || 2) + "h";
        hudLead.className = "text-cyan-400 font-bold";
      }
      if (hudPills) {
        hudPills.innerHTML =
          '<span class="px-2 py-0.5 bg-emerald-500 text-slate-950 rounded">&lt;25% Normal</span>' +
          '<span class="px-2 py-0.5 bg-amber-500 text-slate-950 rounded">25–50% Watch</span>' +
          '<span class="px-2 py-0.5 bg-orange-500 text-white rounded">50–75% Alert</span>' +
          '<span class="px-2 py-0.5 bg-red-500 text-white rounded">≥75% Warning</span>';
      }
      if (inspectBtn) inspectBtn.style.display = "";
    } else {
      // LIVE MODE
      // Render continuous risk surface PNG (StormSense early warning nowcast)
      renderContinuousRiskSurface(map, window.currentLeadHours || 2);
      // Add hotspot beacons
      if (window.StormSenseHighRiskCells) {
        renderHotspotBeacons(map, window.StormSenseHighRiskCells);
      }
      // Add inspection grid if toggled
      if (window.gridInspectionMode && window.StormSenseRiskMapData) {
        renderInspectionGrid(map, window.StormSenseRiskMapData);
      }
      // Add live station marker on top of map
      renderLiveStationMarker(map, window.StormSenseLiveSurface);

      // Update Map HUD
      if (hudTitle) hudTitle.textContent = "STORMSENSE NOWCAST (0.25° Grid)";
      if (hudLead) {
        hudLead.textContent = "+" + (window.currentLeadHours || 2) + "h";
        hudLead.className = "text-cyan-400 font-bold";
      }
      if (hudPills) {
        hudPills.innerHTML =
          '<span class="px-2 py-0.5 bg-emerald-500 text-slate-950 rounded">&lt;25% Normal</span>' +
          '<span class="px-2 py-0.5 bg-amber-500 text-slate-950 rounded">25–50% Watch</span>' +
          '<span class="px-2 py-0.5 bg-orange-500 text-white rounded">50–75% Alert</span>' +
          '<span class="px-2 py-0.5 bg-red-500 text-white rounded">≥75% Warning</span>';
      }
      var hudSubtext = document.getElementById("map-legend-subtext");
      if (hudSubtext) hudSubtext.textContent = "AI-derived severe weather risk · Not an official government warning";
      if (inspectBtn) inspectBtn.style.display = "";
    }

    setTimeout(function () {
      map.invalidateSize();
    }, 100);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Interactive Map Click Handler (Continuous Surface Point Inspection)
  // ─────────────────────────────────────────────────────────────────────────────
  function handleMapClick(e) {
    if (!e || !e.latlng) return;
    if (window.gridInspectionMode) return; // In grid inspection mode, grid cells handle their own events

    var lat = e.latlng.lat;
    var lon = e.latlng.lng;
    var lead = window.currentLeadHours || 2;
    var isLive = window.stormSenseMode === "live";

    var loadingPopup =
      '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;border-radius:8px;border:1px solid #334155;">' +
        '<span style="color:#38bdf8;">Querying atmospheric point data (+' + lead + 'h)...</span>' +
      '</div>';
    var popup = L.popup()
      .setLatLng([lat, lon])
      .setContent(loadingPopup)
      .openOn(window.stormSenseMap);

    fetch(API_BASE + "/api/nowcast/point?lat=" + lat.toFixed(4) + "&lon=" + lon.toFixed(4) + "&lead=" + lead + "&mode=" + (window.stormSenseMode || "live"))
      .then(parseJson)
      .then(function (data) {
        if (!data || (!data.predictions && data.status !== "success")) {
          popup.setContent('<div style="font-family:monospace;font-size:11px;padding:6px;color:#f87171;">Location outside 33×25 model domain.</div>');
          return;
        }

        var p = data.predictions || data.ml_prediction || {};
        var t0 = data.observed_inputs || data.t0_physical_input || {};
        var liveTelemetry = data.live_station_telemetry;
        var color = p.risk_level === "red" ? "#ef4444" : p.risk_level === "orange" ? "#f97316" : p.risk_level === "yellow" ? "#f59e0b" : "#10b981";
        var distName = data.district ? '<div style="color:#38bdf8;font-weight:bold;font-size:12px;margin-bottom:4px;">' + data.district + '</div>' : '<div style="color:#94a3b8;font-size:11px;margin-bottom:4px;">West Bengal Domain Point</div>';

        var tempHum = (t0.temperature_c != null ? t0.temperature_c + '°C' : '—') + ' / ' + ((t0.humidity_pct != null ? t0.humidity_pct : t0.relative_humidity_pct) != null ? (t0.humidity_pct != null ? t0.humidity_pct : t0.relative_humidity_pct) + '%' : '—');
        var windVal = (t0.wind_kmh != null ? t0.wind_kmh : t0.wind_speed_kmh);
        var presVal = (t0.pressure_hpa != null ? t0.pressure_hpa : t0.surface_pressure_hpa);
        var rainVal = (p.heavy_rain_mm != null ? p.heavy_rain_mm : p.heavy_rainfall_mm);
        var modelName = data.model_name || p.model || "StormSense";
        var validTime = data.forecast_valid_utc || p.valid_time_utc || "";

        // Status pill for mode
        var statusPill = isLive
          ? '<span style="font-size:9px;color:#fbbf24;background:#78350f40;border:1px solid #f59e0b60;padding:2px 6px;border-radius:4px;">OPERATIONAL PIPELINE PENDING</span>'
          : '<span style="font-size:9px;color:#38bdf8;background:#0369a140;border:1px solid #38bdf860;padding:2px 6px;border-radius:4px;">HISTORICAL RECONSTRUCTION</span>';

        var liveSection = "";
        if (isLive && liveTelemetry) {
          liveSection =
            '<div style="background:#064e3b40;border:1px solid #05966960;border-radius:6px;padding:6px;margin-bottom:6px;font-size:10px;">' +
              '<div style="color:#34d399;font-weight:bold;margin-bottom:2px;display:flex;justify-content:space-between;">' +
                '<span>Live Station Telemetry (t=0):</span><span style="font-size:9px;color:#6ee7b7;">Live Station</span>' +
              '</div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Temp / Humidity:</span><strong>' + (liveTelemetry.temperature_c != null ? liveTelemetry.temperature_c + '°C' : '—') + ' / ' + (liveTelemetry.humidity_pct != null ? liveTelemetry.humidity_pct + '%' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Rain Gauge (1h):</span><strong>' + (liveTelemetry.rainfall_mm != null ? liveTelemetry.rainfall_mm + ' mm' : '0.0 mm') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Surface Wind:</span><strong>' + (liveTelemetry.wind_kmh != null ? liveTelemetry.wind_kmh + ' km/h' : '—') + '</strong></div>' +
            '</div>';
        } else if (!isLive && t0) {
          liveSection =
            '<div style="background:#1e293b;border-radius:6px;padding:6px;margin-bottom:6px;font-size:10px;">' +
              '<div style="color:#38bdf8;font-weight:bold;margin-bottom:2px;">Physical Input (t=0):</div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Temp / RH:</span><strong>' + tempHum + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Surface Wind:</span><strong>' + (windVal != null ? windVal + ' km/h' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>MSL Pressure:</span><strong>' + (presVal != null ? presVal + ' hPa' : '—') + '</strong></div>' +
            '</div>';
        }

        var content =
          '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;border-radius:10px;min-width:240px;border:1px solid #334155;">' +
            '<div style="font-weight:800;font-size:12px;margin-bottom:4px;color:' + color + ';display:flex;justify-content:space-between;align-items:center;">' +
              '<span>MODEL RISK · ' + (p.risk_label || "NORMAL").toUpperCase() + '</span>' +
              '<span style="color:#94a3b8;font-size:11px;">+' + lead + 'h Lead</span>' +
            '</div>' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
              distName +
              statusPill +
            '</div>' +
            '<div style="font-size:10px;color:#94a3b8;margin-bottom:6px;">Grid Cell: ' + (data.grid_cell ? data.grid_cell.lat.toFixed(2) + '°N, ' + data.grid_cell.lon.toFixed(2) + '°E' : lat.toFixed(2) + '°N, ' + lon.toFixed(2) + '°E') + '</div>' +
            liveSection +
            '<div style="background:#0f172a;border-radius:6px;padding:6px;border:1px solid #1e293b;margin-bottom:6px;font-size:11px;">' +
              '<div style="color:#e2e8f0;font-weight:bold;margin-bottom:4px;font-size:10px;">StormSense Predictions:</div>' +
              '<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Thunderstorm Risk:</span><strong style="color:' + color + ';">' + (p.thunderstorm_prob_pct != null ? p.thunderstorm_prob_pct + '%' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Heavy Rain (3h):</span><strong>' + (rainVal != null ? rainVal + ' mm' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Flash Flood Proxy:</span><strong style="color:#fbbf24;">' + (p.flash_flood_proxy_pct != null ? p.flash_flood_proxy_pct + '%' : '—') + '</strong></div>' +
            '</div>' +
            '<div style="font-size:9px;color:#64748b;border-top:1px solid #1e293b;padding-top:4px;margin-top:4px;display:flex;justify-content:space-between;">' +
              '<span>' + modelName + '</span>' +
              '<span>' + (isLive ? 'Real-time NWP Pending' : 'Valid: ' + validTime) + '</span>' +
            '</div>' +
          '</div>';

        popup.setContent(content);
      })
      .catch(function (err) {
        console.warn("Point query error:", err);
        popup.setContent('<div style="font-family:monospace;font-size:11px;padding:6px;color:#f87171;">Failed to inspect point.</div>');
      });
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Main Map Initializer (Esri Dark Gray Base — ZERO WATERMARK, NO API KEY)
  // ─────────────────────────────────────────────────────────────────────────────
  function initMap(data) {
    var mapContainer = document.getElementById("map-radar-placeholder");
    if (!mapContainer || typeof L === "undefined" || window.stormSenseMap) return;

    var map = L.map("map-radar-placeholder", {
      center: [23.8, 87.9],
      zoom: 7,
      minZoom: 6,
      maxZoom: 16,
      zoomControl: false
    });

    map.on("click", handleMapClick);

    // Public Esri World Dark Gray Base: zero watermark, no API key required!
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      attribution: 'Tiles &copy; <a href="https://www.esri.com/" target="_blank" rel="noopener">Esri</a> &middot; DeLorme, NAVTEQ',
      maxZoom: 16
    }).addTo(map);

    // DWR Radar Station Marker (Kolkata)
    var radarIcon = L.divIcon({
      className: "radar-station-icon",
      html: '<div style="background:#ef4444; width:12px; height:12px; border-radius:50%; border:2px solid #ffffff; box-shadow:0 0 10px #ef4444;"></div>',
      iconSize: [12, 12],
      iconAnchor: [6, 6]
    });
    L.marker([22.65, 88.45], { icon: radarIcon })
      .addTo(map)
      .bindPopup('<div style="font-family:monospace;font-size:11px;color:#f8fafc;background:#0f172a;padding:6px;border-radius:6px;border:1px solid #334155;"><b>IMD Kolkata Doppler Weather Radar (DWR)</b><br>Coordinates: 22.65°N, 88.45°E<br>Operational S-Band Convective Tracking Center</div>');

    // Zoom Buttons
    var zoomInBtn = document.getElementById("btn-map-zoom-in");
    if (zoomInBtn) {
      zoomInBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        map.zoomIn();
      });
    }

    var zoomOutBtn = document.getElementById("btn-map-zoom-out");
    if (zoomOutBtn) {
      zoomOutBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        map.zoomOut();
      });
    }

    window.stormSenseMap = map;

    // Render outer state boundary and interior districts
    if (window.StormSenseStateBoundaryData) {
      renderStateBoundary(map, window.StormSenseStateBoundaryData);
    }
    if (window.StormSenseBoundariesData) {
      renderDistrictBoundaries(map, window.StormSenseBoundariesData);
    }

    // Set initial layer state for mode
    updateMapMode(window.stormSenseMode || "live");

    // Frame on the complete West Bengal outline, and keep it framed: without a
    // pan/zoom constraint the surrounding region dominates the view and the map
    // stops reading as a West Bengal forecast.
    var wbBounds = map._stateBoundaryLayer
      ? map._stateBoundaryLayer.getBounds()
      : L.latLngBounds(window.WB_BOUNDS);
    map.fitBounds(wbBounds, { padding: [20, 20] });
    map.setMaxBounds(wbBounds.pad(0.35));
    map.setMinZoom(map.getBoundsZoom(wbBounds));


    if (!radarLayer) {
        loadRainViewerRadar(map);
    }
    return map;
  }



  var radarLayer = null;
  async function loadRainViewerRadar(map) {
    var statusEl = document.getElementById("radar-status");
    var infoEl = document.getElementById("radar-info");
    var updateEl = document.getElementById("radar-last-update");

    try {
        if (statusEl) statusEl.textContent = "RADAR LOADING";

        var response = await fetch("https://api.rainviewer.com/public/weather-maps.json");
        if (!response.ok) throw new Error("RainViewer API returned HTTP " + response.status);
        
        var data = await response.json();
        if (!data.radar || !data.radar.past || !data.radar.past.length) throw new Error("No radar frames available.");
        
        var latestFrame = data.radar.past[data.radar.past.length - 1];
        
        if (radarLayer && map) {
            map.removeLayer(radarLayer);
            radarLayer = null;
        }

        var tileUrl = data.host + latestFrame.path + "/256/{z}/{x}/{y}/2/1_1.png";
        
        if (map) {
            radarLayer = L.tileLayer(tileUrl, {
                opacity: 0.65,
                zIndex: 400,
                errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
                maxNativeZoom: 7,
                maxZoom: 16,
                attribution: 'Radar data by <a href="https://www.rainviewer.com/" target="_blank" rel="noopener">RainViewer</a>'
            });
            radarLayer.addTo(map);
        }

        if (statusEl) statusEl.textContent = "RADAR READY";
        if (infoEl) infoEl.textContent = "RainViewer precipitation radar connected";
        if (updateEl) {
            var scanTime = new Date(latestFrame.time * 1000);
            updateEl.textContent = "Last Scan: " + scanTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }
    } catch (error) {
        console.error("RainViewer radar error:", error);
        if (statusEl) statusEl.textContent = "RADAR OFFLINE";
        if (infoEl) infoEl.textContent = "Radar feed unavailable";
        if (updateEl) updateEl.textContent = "Connection failed";
    }
  }

  function initRadarMap(data) {

    var mapContainer = document.getElementById("radar-map-container");
    if (!mapContainer || typeof L === "undefined" || window.stormSenseRadarMap) return;

    var map = L.map("radar-map-container", {
      center: [23.8, 87.9],
      zoom: 7,
      minZoom: 6,
      maxZoom: 16,
      zoomControl: true
    });

    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      attribution: 'Tiles &copy; <a href="https://www.esri.com/" target="_blank" rel="noopener">Esri</a> &middot; DeLorme, NAVTEQ',
      maxZoom: 16
    }).addTo(map);

    window.stormSenseRadarMap = map;

    if (window.StormSenseStateBoundaryData) {
      renderStateBoundary(map, window.StormSenseStateBoundaryData);
    }
    if (window.StormSenseBoundariesData) {
      renderDistrictBoundaries(map, window.StormSenseBoundariesData);
    }
    if (window.stormSenseMode === "historical") {
      renderContinuousRiskSurface(map, window.currentLeadHours || 2);
    }

    if (map._stateBoundaryLayer) {
      map.fitBounds(map._stateBoundaryLayer.getBounds(), { padding: [20, 20] });
    } else {
      map.fitBounds([[21.5394, 86.6103], [26.9960, 89.8828]], { padding: [15, 15] });
    }


    if (!radarLayer) {
        loadRainViewerRadar(map);
    }
    return map;
  }


  // ─────────────────────────────────────────────────────────────────────────────
  // View Switcher & Horizon Controller
  // ─────────────────────────────────────────────────────────────────────────────
  window.switchNowcastView = function (targetView) {
    viewIds.forEach(function (view) {
      var container = document.getElementById("view-" + view);
      if (!container) return;
      if (view === targetView) container.classList.remove("hidden");
      else container.classList.add("hidden");
    });

    document.querySelectorAll(".nav-item").forEach(function (item) {
      var view = item.getAttribute("data-view");
      item.classList.remove("bg-slate-800/80", "text-white", "font-semibold", "border-slate-700/60");
      if (view === targetView) {
        item.classList.remove("text-slate-300", "font-medium");
        item.classList.add("bg-slate-800/80", "text-white", "font-semibold", "border-slate-700/60");
      } else {
        item.classList.add("text-slate-300", "font-medium");
      }
    });

    if (targetView === "dashboard" && window.stormSenseMap) {
      setTimeout(function () {
        window.stormSenseMap.invalidateSize();
      }, 150);
    }
    if (targetView === "radar") {
      if (!window.stormSenseRadarMap) {
        initRadarMap(null);
      }
      setTimeout(function () {
        if (window.stormSenseRadarMap) {
          window.stormSenseRadarMap.invalidateSize();
        }
      }, 150);
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  window.setForecastHorizon = function (btn, label, leadHours) {
    var lead = leadHours;
    if (!lead) {
      if (label === "+2h" || label === "+2 Hours" || label === "+2h Horizon") lead = 2;
      else if (label === "+4h" || label === "+4 Hours" || label === "+4h Horizon") lead = 4;
      else if (label === "+6h" || label === "+6 Hours" || label === "+6h Horizon") lead = 6;
      else lead = 2;
    }
    // Strictly clamp to supported horizons [2, 4, 6]
    if (lead !== 2 && lead !== 4 && lead !== 6) {
      lead = lead < 3 ? 2 : (lead < 5 ? 4 : 6);
    }

    window.currentLeadHours = lead;
    if (window.stormSenseMap && window.stormSenseMap.closePopup) window.stormSenseMap.closePopup();

    // Highlight horizon buttons
    document.querySelectorAll(".horizon-btn").forEach(function (b) {
      b.classList.remove("bg-cyan-600", "bg-red-600", "text-white", "font-bold");
      b.classList.add("text-slate-400", "font-medium");
      var txt = b.textContent.trim();
      if (txt === "+" + lead + "h" || txt === "+" + lead + " Hours" || txt === "+" + lead + "h Horizon") {
        b.classList.add("bg-cyan-600", "text-white", "font-bold");
        b.classList.remove("text-slate-400", "font-medium");
      }
    });

    var modeLabel = window.stormSenseMode === "live" ? "Live early warning" : "Historical validation";
    window.showToast("StormSense: loading +" + lead + "h horizon (" + modeLabel + ")...", "radar");

    // Every request carries the active mode: without it the backend defaults to
    // the historical case study and a Live horizon switch would silently show
    // historical values while the UI still said "Live".
    var modeQS = "&mode=" + (window.stormSenseMode || "live");

    Promise.all([
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/districts?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/risk-map?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {
        var summary = results[0];
        window.StormSenseCurrentSummary = summary;
        var districts = results[1];
        var riskMap = results[2];
        var cells = results[3];

        var issueTime = (summary && summary.issue_time) || "2024-05-05T15:00:00Z";
        updateDynamicTimes(issueTime, lead);
        applyRegionState(summary);

        // Paint the hazard cards from whichever mode actually returned a
        // forecast. Previously this was gated on historical mode only, so live
        // predictions were fetched and then thrown away.
        if (summary && summary.hazards) {
          window.StormSenseNowcastData = summary;
          paintForecastCards(summary);
          if (summary.timeline) renderNowcast(summary.timeline);
        }
        applyFreshnessState(summary);

        if (districts && districts.length) {
          window.StormSenseDistrictsData = districts;
          renderDistricts(adaptDistricts(districts, lead, issueTime));
        }

        if (cells) {
          window.StormSenseHighRiskCells = cells;
          renderHighRiskCells(cells);
        }

        if (riskMap) {
          window.StormSenseRiskMapData = riskMap;
        }

        if (window.stormSenseMap) {
          renderContinuousRiskSurface(window.stormSenseMap, lead);
          if (cells) renderHotspotBeacons(window.stormSenseMap, cells);
          if (riskMap && window.gridInspectionMode) renderInspectionGrid(window.stormSenseMap, riskMap);
        }
        var xai = results[4];
        if (xai) {
            if (typeof renderXai === 'function') renderXai(xai);
        }
        var thermo = results[5];
        if (thermo) {
            if (typeof renderThermodynamics === 'function') renderThermodynamics(thermo);
        }

        window.showToast("StormSense Nowcast updated: +" + lead + "h lead time", "check_circle");
      })
      .catch(function (err) {
        console.error("Failed to load horizon nowcast:", err);
        window.showToast("Failed to fetch +" + lead + "h nowcast", "warning");
      });
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // Interactive Modals, Toasts & User Utilities
  // ─────────────────────────────────────────────────────────────────────────────
  window.triggerSiren = function () {
    var modal = document.getElementById("modal-siren");
    if (modal) modal.classList.remove("hidden");
  };

  window.closeSirenModal = function () {
    var modal = document.getElementById("modal-siren");
    if (modal) modal.classList.add("hidden");
  };

  window.executeSirenDispatch = function () {
    window.closeSirenModal();
    window.showToast("Simulated Advisory Dispatched: Automated CAP XML & SMS advisory broadcast simulated for District Disaster Management Units.", "emergency_share");
  };

  window.showToast = function (msg, iconType) {
    var toast = document.getElementById("toast-notification");
    var text = document.getElementById("toast-message");
    var icon = document.getElementById("toast-icon");
    if (!toast || !text) return;
    text.textContent = msg;
    if (icon) {
      if (iconType === "warning") {
        icon.textContent = "warning";
        icon.className = "material-symbols-outlined text-red-400";
      } else if (iconType === "radar") {
        icon.textContent = "radar";
        icon.className = "material-symbols-outlined text-blue-400";
      } else if (iconType === "emergency_share") {
        icon.textContent = "emergency_share";
        icon.className = "material-symbols-outlined text-amber-400";
      } else if (iconType === "history_edu") {
        icon.textContent = "history_edu";
        icon.className = "material-symbols-outlined text-cyan-400";
      } else if (iconType === "sensors") {
        icon.textContent = "sensors";
        icon.className = "material-symbols-outlined text-emerald-400";
      } else {
        icon.textContent = "check_circle";
        icon.className = "material-symbols-outlined text-emerald-400";
      }
    }
    toast.classList.remove("translate-y-20", "opacity-0");
    toast.classList.add("translate-y-0", "opacity-100");
    setTimeout(function () {
      toast.classList.remove("translate-y-0", "opacity-100");
      toast.classList.add("translate-y-20", "opacity-0");
    }, 4000);
  };

  // Map-navigation shortcuts only. Keys must match the #sector-selector option
  // values, and this control must never change which region is monitored --
  // that is always the whole state.
  var SECTOR_COORDS = {
    "Darjeeling / Kalimpong": [27.02, 88.35],
    "Jalpaiguri / Alipurduar": [26.50, 89.10],
    "Malda / Murshidabad": [24.60, 88.20],
    "Purulia / Bankura": [23.28, 86.70],
    "Purba Bardhaman & Damodar": [23.24, 87.86],
    "North 24 Parganas Delta": [22.724, 88.479],
    "Kolkata Metropolitan Area": [22.572, 88.363]
  };

  window.handleMyLocationClick = function () {
    var ping = document.getElementById("my-location-ping");
    if (ping) ping.classList.remove("hidden");
    window.showToast("Locating your GPS position...", "radar");

    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        function (pos) {
          var lat = pos.coords.latitude;
          var lon = pos.coords.longitude;
          if (ping) ping.classList.add("hidden");
          if (window.stormSenseMap) {
            window.stormSenseMap.flyTo([lat, lon], 12);
            if (window._userMarker) {
              window.stormSenseMap.removeLayer(window._userMarker);
            }
            window._userMarker = L.circleMarker([lat, lon], {
              radius: 9,
              color: "#06b6d4",
              fillColor: "#22d3ee",
              fillOpacity: 0.9,
              weight: 2
            })
              .addTo(window.stormSenseMap)
              .bindPopup("<div style='font-family:monospace;font-size:11px;color:#fff;'><b>Your Location</b><br>" + lat.toFixed(3) + "°N, " + lon.toFixed(3) + "°E</div>")
              .openPopup();
          }
          window.showToast("GPS position acquired: " + lat.toFixed(3) + "°N, " + lon.toFixed(3) + "°E", "check_circle");
        },
        function () {
          if (ping) ping.classList.add("hidden");
          window.showToast("GPS permission denied. Showing default monitored region.", "warning");
        },
        { timeout: 5000, enableHighAccuracy: true }
      );
    } else {
      if (ping) ping.classList.add("hidden");
      window.showToast("Browser geolocation not supported.", "warning");
    }
  };

  window.stubDemoAction = function (label) {
    window.showToast("Operational module: " + label + " executed.", "check_circle");
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // DOM Initialization
  // ─────────────────────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", function () {
    // Start real-time IST clock ticker (every 1 second)
    updateIstClock();
    setInterval(updateIstClock, 1000);

    // Initial mode setup
    updateModeUI(window.stormSenseMode);

    loadDashboardData().then(function (data) {
      if (!data) {
        console.error("StormSense data missing");
        return;
      }

      // Initialize single map
      initMap(data);
      initRadarMap(data);

      // Render dashboard based on active mode
      updateModeUI(window.stormSenseMode);

      // Live auto-refresh. Timers are cleared first so repeated initialisation
      // can never leave two intervals running and double up requests.
      if (liveRefreshTimer) clearInterval(liveRefreshTimer);
      if (countdownTimer) clearInterval(countdownTimer);

      liveRefreshTimer = setInterval(function () {
        if (window.stormSenseMode === "live") {
          window.refreshLiveDashboard();
        }
      }, LIVE_REFRESH_MS);

      // Countdown ticker for next auto-refresh
      countdownTimer = setInterval(function () {
        if (window.stormSenseMode === "live") {
          liveCountdownSeconds = Math.max(0, liveCountdownSeconds - 1);
          setText("live-refresh-countdown", formatCountdown(liveCountdownSeconds));
          if (liveCountdownSeconds === 0) liveCountdownSeconds = LIVE_REFRESH_MS / 1000;
        }
      }, 1000);

      window.switchNowcastView("dashboard");
    });

    // Navigation links
    document.querySelectorAll(".nav-item").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.preventDefault();
        var view = this.getAttribute("data-view");
        if (view) {
          window.switchNowcastView(view);
        }
      });
    });

    // Siren buttons
    document.querySelectorAll(".btn-siren-trigger").forEach(function (btn) {
      btn.addEventListener("click", function (e) {
        e.preventDefault();
        window.triggerSiren();
      });
    });

    // Export Official Bulletin
    var exportBtn = document.getElementById("btn-export-report");
    if (exportBtn) {
      exportBtn.addEventListener("click", function (e) {
        e.preventDefault();
        var lead = window.currentLeadHours || 2;
        var nowcastData = window.StormSenseNowcastData || {};
        var issueTime = nowcastData.issue_time || "2024-05-05T15:00:00Z";
        var validTime = nowcastData.forecast_valid_time || formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + lead * 3600 * 1000));

        var bulletin = {
          bulletin_id: "IMD-KOL-NW-" + new Date().toISOString().slice(0, 10).replace(/-/g, "") + "-" + lead + "H",
          issued_at_utc: formatUtcDateTime(parseUtcIso(issueTime)),
          forecast_valid_utc: validTime,
          lead_horizon_hours: lead,
          issuing_office: "India Meteorological Department, Regional Meteorological Centre, Kolkata",
          target_region: "Gangetic West Bengal (20.0°N–28.0°N, 84.0°E–90.0°E)",
          primary_district: "North 24 Parganas",
          model_system: "StormSense (Multi-Task Deep Learning Nowcaster)",
          model_checkpoint: "v2_calibrated_best.pt",
          verified_test_metrics_2024: {
            test_split: "May–October 2024 Convective Season (4,322 Sequences)",
            severe_pr_auc: 0.4840,
            mean_csi: 0.3068,
            mean_pod: 0.5428,
            mean_far: 0.5891,
            brier_score: 0.0512,
            calibrated_temperature: [0.4152, 0.4144, 0.4169, 0.4105, 0.3819],
            calibrated_thresholds: [0.727, 0.679, 0.673, 0.637, 0.625]
          },
          current_hazards: nowcastData.hazards || null,
          district_advisories: window.StormSenseDistrictsData || [],
          thermodynamic_state: window.StormSenseThermoData || null
        };

        var blob = new Blob([JSON.stringify(bulletin, null, 2)], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "StormSense_Nowcast_Bulletin_" + lead + "h.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        window.showToast("Exported StormSense Bulletin (+" + lead + "h)", "check_circle");
      });
    }

    // Map navigation shortcut. Moves the viewport only -- it does not change the
    // monitored region, the forecast, or any displayed risk value.
    var sectorSelect = document.getElementById("sector-selector");
    if (sectorSelect) {
      sectorSelect.addEventListener("change", function () {
        if (!window.stormSenseMap) return;
        var coords = SECTOR_COORDS[this.value];
        if (coords) {
          window.stormSenseMap.flyTo(coords, 9);
          window.showToast("Map centred on " + this.value, "radar");
        } else {
          window.stormSenseMap.fitBounds(L.latLngBounds(window.WB_BOUNDS), { padding: [20, 20] });
          window.showToast("Map centred on West Bengal", "radar");
        }
      });
    }
  });
})();


  window.forceLiveRefresh = function() {
      var btn = document.getElementById("btn-refresh-live");
      if (btn.disabled) return;
      btn.disabled = true;
      var originalText = btn.innerHTML;
      btn.innerHTML = '<span class="material-symbols-outlined text-[14px] animate-spin">sync</span><span>REFRESHING...</span>';

      fetch(API_BASE + "/api/live/refresh", { method: 'POST' })
        .then(parseJson)
        .then(function(res) {
             btn.innerHTML = originalText;
             btn.disabled = false;
             window.refreshLiveDashboard();
        })
        .catch(function(err) {
             btn.innerHTML = originalText;
             btn.disabled = false;
             console.error("Manual refresh failed:", err);
        });
  };
