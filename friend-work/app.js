/**
 * StormSense Phase 1 Frontend
 * Live sector-aware dashboard using FastAPI, OpenWeather and Open-Meteo.
 *
 * Backend endpoints used:
 *   GET /api/stormsense/timeline?sector=<sector>
 *   GET /api/stormsense/nowcast?sector=<sector>
 */

(function () {
  "use strict";

  // ============================================================
  // CONFIGURATION
  // ============================================================

  var API_BASE = "http://127.0.0.1:8000";

  var viewIds = [
    "dashboard",
    "radar",
    "advisories",
    "wrf",
    "xai",
    "gis",
    "threshold",
    "streams"
  ];

  var currentSector = "default";

  var DEFAULT_LOCATION = {
    lat: 22.724,
    lon: 88.479,
    district: "North 24 Parganas, West Bengal"
  };

  // ============================================================
  // RADAR STATE
  // ============================================================

  var radarMap = null;
  var radarLayer = null;
  var radarInitialized = false;
  var radarRefreshTimer = null;

// ============================================================
// RAINVIEWER RADAR
// ============================================================

async function loadRainViewerRadar(map) {
    var statusEl = document.getElementById("radar-status");
    var infoEl = document.getElementById("radar-info");
    var updateEl = document.getElementById("radar-last-update");

    try {
        if (statusEl) {
            statusEl.textContent = "RADAR LOADING";
        }

        var response = await fetch(
            "https://api.rainviewer.com/public/weather-maps.json"
        );

        if (!response.ok) {
            throw new Error(
                "RainViewer API returned HTTP " + response.status
            );
        }

        var data = await response.json();

        console.log("RainViewer radar metadata:", data);

        if (
            !data.radar ||
            !data.radar.past ||
            !data.radar.past.length
        ) {
            throw new Error("No radar frames available.");
        }

        var latestFrame =
            data.radar.past[data.radar.past.length - 1];

        console.log("Latest radar frame:", latestFrame);

        /*
         * Remove the previous radar layer if one exists.
         */
        if (radarLayer && map) {
            map.removeLayer(radarLayer);
            radarLayer = null;
        }

        /*
         * Build the RainViewer tile URL dynamically.
         */
        var tileUrl =
            data.host +
            latestFrame.path +
            "/512/{z}/{x}/{y}/2/1_1.png";

        console.log(
            "RainViewer tile URL:",
            tileUrl
        );

        /*
         * Add radar precipitation layer.
         */
        if (map) {
            radarLayer = L.tileLayer(
                tileUrl,
                {
                    opacity: 0.70,
                    maxNativeZoom: 7,
                    maxZoom: 18,
                    attribution:
                        'Weather data by <a href="https://www.rainviewer.com/" target="_blank" rel="noopener">RainViewer</a>'
                }
            );

            radarLayer.addTo(map);
        }

        /*
         * Update UI status.
         */
        if (statusEl) {
            statusEl.textContent = "RADAR READY";
        }

        if (infoEl) {
            infoEl.textContent =
                "RainViewer precipitation radar connected";
        }

        if (updateEl) {
            var scanTime =
                new Date(
                    latestFrame.time * 1000
                );

            updateEl.textContent =
                "Last Scan: " +
                scanTime.toLocaleTimeString(
                    [],
                    {
                        hour: "2-digit",
                        minute: "2-digit"
                    }
                );
        }

    } catch (error) {

        console.error(
            "RainViewer radar error:",
            error
        );

        if (statusEl) {
            statusEl.textContent =
                "RADAR OFFLINE";
        }

        if (infoEl) {
            infoEl.textContent =
                "Radar feed unavailable";
        }

        if (updateEl) {
            updateEl.textContent =
                "Connection failed";
        }
    }
}


  // ============================================================
  // DOM HELPERS
  // ============================================================

  function setText(id, value) {
    var el = document.getElementById(id);

    if (!el || value == null) {
      return;
    }

    el.textContent = String(value);
  }

  function setWidth(id, pct) {
    var el = document.getElementById(id);

    if (!el) {
      return;
    }

    var value = Number(pct);

    if (!isFinite(value)) {
      value = 0;
    }

    value = Math.max(0, Math.min(100, value));

    el.style.width = value + "%";
  }

  function safeNumber(value, fallback) {
    var n = Number(value);

    if (!isFinite(n)) {
      return fallback == null ? 0 : fallback;
    }

    return n;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ============================================================
  // HAZARD HELPERS
  // ============================================================

  function getHazardProbability(hazard) {
    if (!hazard) {
      return 0;
    }

    if (hazard.probability != null) {
      return Math.round(
        safeNumber(hazard.probability, 0)
      );
    }

    if (hazard.probabilityPct != null) {
      return Math.round(
        safeNumber(hazard.probabilityPct, 0)
      );
    }

    return 0;
  }

  function probabilityToLevel(probability) {
    var p = safeNumber(probability, 0);

    if (p >= 75) {
      return "red";
    }

    if (p >= 50) {
      return "orange";
    }

    if (p >= 25) {
      return "yellow";
    }

    return "green";
  }

  function getHazardTrend(probability) {
    var p = safeNumber(probability, 0);

    if (p >= 75) {
      return "HIGH RISK";
    }

    if (p >= 50) {
      return "ELEVATED";
    }

    if (p >= 25) {
      return "MONITOR";
    }

    return "LOW";
  }

  function getHazardDetail(hazardName, probability) {
    var p = safeNumber(probability, 0);

    if (p >= 75) {
      return (
        hazardName +
        " conditions indicate high short-term risk."
      );
    }

    if (p >= 50) {
      return (
        hazardName +
        " risk is elevated within the 0–6h nowcast window."
      );
    }

    if (p >= 25) {
      return (
        hazardName +
        " conditions require continued monitoring."
      );
    }

    return (
      hazardName +
      " risk currently remains low."
    );
  }

  function adaptHazard(hazard, hazardName) {
    var probability =
      getHazardProbability(hazard);

    var level =
      hazard && hazard.level
        ? String(hazard.level).toLowerCase()
        : probabilityToLevel(probability);

    return {
      probabilityPct: probability,
      level: level,
      trendLabel: getHazardTrend(probability),
      detail: getHazardDetail(
        hazardName,
        probability
      )
    };
  }

  function getOverallAction(level) {
    var l = String(level || "").toLowerCase();

    if (l === "red") {
      return "TAKE IMMEDIATE ACTION";
    }

    if (l === "orange") {
      return "PREPARE FOR HAZARD";
    }

    if (l === "yellow") {
      return "MONITOR CONDITIONS";
    }

    return "NORMAL MONITORING";
  }

  function getOverallStage(level) {
    var l = String(level || "").toLowerCase();

    if (l === "red") {
      return "CRITICAL";
    }

    if (l === "orange") {
      return "ALERT";
    }

    if (l === "yellow") {
      return "WATCH";
    }

    return "NORMAL";
  }

  // ============================================================
  // WIND DIRECTION
  // ============================================================

  function getWindDirection(degrees) {
    if (
      degrees == null ||
      isNaN(Number(degrees))
    ) {
      return "";
    }

    var directions = [
      "N",
      "NNE",
      "NE",
      "ENE",
      "E",
      "ESE",
      "SE",
      "SSE",
      "S",
      "SSW",
      "SW",
      "WSW",
      "W",
      "WNW",
      "NW",
      "NNW"
    ];

    var index =
      Math.round(Number(degrees) / 22.5) % 16;

    return directions[index];
  }

  // ============================================================
  // HAZARD COLORS
  // ============================================================

  function levelClass(level) {
    var l =
      String(level || "green").toLowerCase();

    if (l === "red") {
      return {
        text: "text-red-400",
        bg: "bg-red-500/15",
        border: "border-red-500/40",
        bar: "bg-red-500",
        pill: "bg-red-600 text-white"
      };
    }

    if (l === "orange") {
      return {
        text: "text-orange-400",
        bg: "bg-orange-500/15",
        border: "border-orange-500/40",
        bar: "bg-orange-500",
        pill: "bg-orange-500 text-white"
      };
    }

    if (l === "yellow") {
      return {
        text: "text-amber-400",
        bg: "bg-amber-500/15",
        border: "border-amber-500/40",
        bar: "bg-amber-500",
        pill: "bg-amber-500 text-slate-950"
      };
    }

    return {
      text: "text-emerald-400",
      bg: "bg-emerald-500/15",
      border: "border-emerald-500/40",
      bar: "bg-emerald-500",
      pill: "bg-emerald-500/20 text-emerald-300"
    };
  }

  // ============================================================
  // LIVE DATA FETCH
  // ============================================================

  function loadDashboardData(sector) {
    var selectedSector =
      sector ||
      currentSector ||
      "default";

    console.log(
      "StormSense: requesting live data for sector:",
      selectedSector
    );

    var query =
      "?sector=" +
      encodeURIComponent(selectedSector);

    var timelineUrl =
      API_BASE +
      "/api/stormsense/timeline" +
      query;

    var nowcastUrl =
      API_BASE +
      "/api/stormsense/nowcast" +
      query;

      var districtUrl =
  API_BASE +
  "/api/district-advisories" +
  query;

    console.log(
      "StormSense timeline URL:",
      timelineUrl
    );

    console.log(
      "StormSense nowcast URL:",
      nowcastUrl
    );

    return Promise.all([
      fetch(timelineUrl, {
        method: "GET",
        cache: "no-store",
        headers: {
          Accept: "application/json"
        }
      }),

      fetch(nowcastUrl, {
        method: "GET",
        cache: "no-store",
        headers: {
          Accept: "application/json"
        }
      }),

      fetch(districtUrl, {
  method: "GET",
  cache: "no-store",
  headers: {
    Accept: "application/json"
  }
})
    ])
      .then(function (responses) {
        var timelineResponse =
          responses[0];

        var nowcastResponse =
          responses[1];

        var districtResponse =
          responses[2];

        console.log(
          "StormSense timeline HTTP:",
          timelineResponse.status
        );

        console.log(
          "StormSense nowcast HTTP:",
          nowcastResponse.status
        );

        if (!timelineResponse.ok) {
          throw new Error(
            "Timeline API returned HTTP " +
              timelineResponse.status
          );
        }

        if (!nowcastResponse.ok) {
          throw new Error(
            "Nowcast API returned HTTP " +
              nowcastResponse.status
          );
        }

        if (!districtResponse.ok) {
            console.warn(
                "District Advisory API returned HTTP " +
                  districtResponse.status
          );
        }

        return Promise.all([
  timelineResponse.json(),
  nowcastResponse.json(),
  districtResponse.ok
    ? districtResponse.json()
    : Promise.resolve({
        districts: []
      })
]);
      })
      .then(function (results) {
        var timelineData =
          results[0];

        var nowcastData =
          results[1];

        var districtData =
          results[2];

        console.log(
          "StormSense timeline response:",
          timelineData
        );

        console.log(
          "StormSense nowcast response:",
          nowcastData
        );

        if (
          !timelineData ||
          typeof timelineData !== "object"
        ) {
          throw new Error(
            "Invalid timeline JSON response."
          );
        }

        if (
          !nowcastData ||
          typeof nowcastData !== "object"
        ) {
          throw new Error(
            "Invalid nowcast JSON response."
          );
        }

        currentSector =
          nowcastData.sector ||
          timelineData.sector ||
          selectedSector;

        return mergeLiveWeatherData(
          timelineData,
          nowcastData,
          districtData
        );
      })
      .catch(function (error) {
        console.error(
          "StormSense live API unavailable:",
          error
        );

        throw error;
      });
  }

  // ============================================================
  // MERGE LIVE DATA
  // IMPORTANT:
  // Live backend data is now PRIMARY.
  // Mock data is OPTIONAL only.
  // ============================================================

  function mergeLiveWeatherData(
    timelineData,
    nowcastData,
    districtData
  ) {
    if (
      !timelineData ||
      !nowcastData
    ) {
      console.error(
        "StormSense: missing API response."
      );

      return null;
    }

    if (
      !Array.isArray(
        timelineData.timeline
      )
    ) {
      console.error(
        "StormSense: timeline array missing:",
        timelineData
      );

      return null;
    }

    var timeline =
      timelineData.timeline;

    var currentTimelinePoint =
      timeline.length > 0
        ? timeline[0]
        : {};

    var backendCurrent =
      nowcastData.current_conditions ||
      {};

    // ----------------------------------------------------------
    // STORE RAW LIVE DATA
    // ----------------------------------------------------------

    window.StormSenseLiveData =
      timelineData;

    window.StormSenseNowcastData =
      nowcastData;

    // ----------------------------------------------------------
    // OPTIONAL MOCK DATA
    // ----------------------------------------------------------

    var mock =
      window.StormSenseMockData;

    var merged;

    if (mock) {
      try {
        merged = JSON.parse(
          JSON.stringify(mock)
        );
      } catch (error) {
        console.warn(
          "StormSense: unable to clone mock data. Using live fallback.",
          error
        );

        merged = {};
      }
    } else {
      merged = {};
    }

    // ----------------------------------------------------------
    // GUARANTEED LIVE STRUCTURE
    // ----------------------------------------------------------

    merged.location =
      merged.location || {};

    merged.currentConditions =
      merged.currentConditions || {};

    merged.systemStatus =
      merged.systemStatus || {};

    merged.hazards =
      merged.hazards || {};

    // ----------------------------------------------------------
    // TIMELINE
    // ----------------------------------------------------------

    merged.timeline = timeline;

    // ----------------------------------------------------------
    // LOCATION
    // Backend provides latitude / longitude directly.
    // ----------------------------------------------------------

    merged.location.lat =
      safeNumber(
        nowcastData.latitude != null
          ? nowcastData.latitude
          : timelineData.latitude,
        DEFAULT_LOCATION.lat
      );

    merged.location.lon =
      safeNumber(
        nowcastData.longitude != null
          ? nowcastData.longitude
          : timelineData.longitude,
        DEFAULT_LOCATION.lon
      );

    merged.location.district =
      nowcastData.location ||
      timelineData.location ||
      DEFAULT_LOCATION.district;

    merged.location.shortLabel =
      merged.location.district;

    merged.location.state =
      "West Bengal";

    merged.location.country =
      "India";

    // ----------------------------------------------------------
    // CURRENT WEATHER
    //
    // PRIMARY SOURCE:
    // nowcastData.current_conditions
    //
    // FALLBACK:
    // timelineData.timeline[0]
    // ----------------------------------------------------------

    var temperature =
      backendCurrent.temperature != null
        ? backendCurrent.temperature
        : currentTimelinePoint.temperature;

    var humidity =
      backendCurrent.humidity != null
        ? backendCurrent.humidity
        : currentTimelinePoint.humidity;

    var windSpeed =
      backendCurrent.wind_speed != null
        ? backendCurrent.wind_speed
        : currentTimelinePoint.wind_speed;

    var windDirection =
      currentTimelinePoint.wind_direction;

    if (
      windDirection == null &&
      backendCurrent.wind_direction != null
    ) {
      windDirection =
        backendCurrent.wind_direction;
    }

    var rainfall =
      backendCurrent.rainfall_1h != null
        ? backendCurrent.rainfall_1h
        : currentTimelinePoint.rainfall_mm;

    merged.currentConditions.temperatureC =
      safeNumber(temperature, 0);

    merged.currentConditions.humidityPct =
      safeNumber(humidity, 0);

    merged.currentConditions.windKmh =
      safeNumber(windSpeed, 0);

    merged.currentConditions.windDirection =
      getWindDirection(windDirection);

    merged.currentConditions.rainfallMmHr =
      safeNumber(rainfall, 0);

    merged.currentConditions.pressureHpa =
      safeNumber(
        backendCurrent.pressure,
        0
      );

    merged.currentConditions.weather =
      backendCurrent.weather ||
      currentTimelinePoint.weather ||
      "";

    // ----------------------------------------------------------
    // LIVE SYSTEM STATUS
    // ----------------------------------------------------------

    var timestamp =
      new Date();

    merged.systemStatus.timestampDisplay =
      timestamp.toLocaleString(
        "en-IN",
        {
          timeZone: "Asia/Kolkata",
          day: "2-digit",
          month: "short",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false
        }
      ) +
      " IST";

    merged.systemStatus.freshnessLabel =
      "LIVE · OpenWeather";

    merged.systemStatus.clusterLabel =
      "Sector: " +
      (
        nowcastData.sector ||
        timelineData.sector ||
        currentSector
      );

    // ----------------------------------------------------------
    // LIVE HAZARDS
    // ----------------------------------------------------------

    var liveHazards =
      nowcastData.hazards || {};

    if (
      liveHazards.thunderstorm
    ) {
      merged.hazards.thunderstorm =
        adaptHazard(
          liveHazards.thunderstorm,
          "Thunderstorm"
        );
    }

    var heavyRain =
      liveHazards.heavy_rainfall ||
      liveHazards.heavyRainfall;

    if (heavyRain) {
      merged.hazards.heavyRainfall =
        adaptHazard(
          heavyRain,
          "Heavy Rainfall"
        );
    }

    var flashFlood =
      liveHazards.flash_flood ||
      liveHazards.flashFlood;

    if (flashFlood) {
      merged.hazards.flashFlood =
        adaptHazard(
          flashFlood,
          "Flash Flood"
        );
    }

    if (liveHazards.overall) {
      var overall =
        liveHazards.overall;

      var overallProbability =
        getHazardProbability(
          overall
        );

      var overallLevel =
        overall.level ||
        probabilityToLevel(
          overallProbability
        );

      merged.hazards.overall = {
        probabilityPct:
          overallProbability,

        level:
          String(
            overallLevel
          ).toLowerCase(),

        actionLabel:
          getOverallAction(
            overallLevel
          ),

        stageLabel:
          getOverallStage(
            overallLevel
          )
      };
    }

    // ----------------------------------------------------------
    // XAI
    // ----------------------------------------------------------

    if (nowcastData.xai) {
      merged.xai =
        nowcastData.xai;

      if (
        nowcastData.xai_inputs
      ) {
        merged.xai.inputs =
          nowcastData.xai_inputs;
      }

      merged.xaiSource =
        "LIVE · Open-Meteo · Rule-based attribution";
    }

    // ----------------------------------------------------------
    // DISTRICTS
    //
    // Backend currently does not return district comparison
    // data, so don't let that block the dashboard.
    // ----------------------------------------------------------

    // ----------------------------------------------------------
// DISTRICT ADVISORIES
// ----------------------------------------------------------

if (
  districtData &&
  Array.isArray(
    districtData.districts
  )
) {
  merged.districts =
    districtData.districts;
} else if (
  !Array.isArray(
    merged.districts
  )
) {
  merged.districts = [];
}

    // ----------------------------------------------------------
    // RADAR PLACEHOLDER
    // ----------------------------------------------------------

    if (
      !merged.radarPlaceholder
    ) {
      merged.radarPlaceholder = {
        hotspots: []
      };
    }

    // ----------------------------------------------------------
    // DISCLAIMER
    // ----------------------------------------------------------

    merged.disclaimer =
      "Live OpenWeather observations connected. Hazard probabilities are generated by the StormSense rule-based Hazard Engine.";

    // ----------------------------------------------------------
    // IDENTIFIERS
    // ----------------------------------------------------------

    merged.sector =
      nowcastData.sector ||
      timelineData.sector ||
      currentSector;

    merged.source =
      nowcastData.source ||
      "OpenWeather + Open-Meteo";

    return merged;
  }

  // ============================================================
  // HAZARD CARD
  // ============================================================

  function applyHazardCard(
    prefix,
    hazard
  ) {
    if (!hazard) {
      return;
    }

    var cls =
      levelClass(
        hazard.level
      );

    setText(
      prefix + "-value",
      getHazardProbability(hazard) +
        "%"
    );

    setText(
      prefix + "-level",
      String(
        hazard.level || "green"
      ).toUpperCase()
    );

    setText(
      prefix + "-trend",
      hazard.trendLabel || ""
    );

    setText(
      prefix + "-detail",
      hazard.detail || ""
    );

    var pill =
      document.getElementById(
        prefix + "-level"
      );

    if (pill) {
      pill.className =
        "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider " +
        cls.bg +
        " " +
        cls.text +
        " " +
        cls.border;
    }

    setWidth(
      prefix + "-bar",
      getHazardProbability(hazard)
    );
  }

  // ============================================================
  // LIVE BULLETINS
  // ============================================================

  function renderBulletins(
    bulletins
  ) {
    var root =
      document.getElementById(
        "active-bulletins-list"
      );

    if (!root) {
      return;
    }

    if (
      !Array.isArray(bulletins) ||
      bulletins.length === 0
    ) {
      root.innerHTML =
        '<div class="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-xs text-slate-400 font-mono">' +
        "No active StormSense advisories at this time." +
        "</div>";

      return;
    }

    root.innerHTML =
      bulletins
        .map(function (b) {
          var level =
            String(
              b.level || "green"
            ).toLowerCase();

          var cls =
            levelClass(level);

          var title =
            b.title ||
            "Meteorological Advisory";

          var body =
            b.body ||
            "No advisory details available.";

          var validUntil =
            b.validUntil ||
            "Next 6 hours";

          var issuer =
            b.issuer ||
            "StormSense Hazard Engine";

          return (
            '<div class="p-4 rounded-xl ' +
            cls.bg +
            ' border ' +
            cls.border +
            ' relative overflow-hidden">' +

              '<div class="flex items-start gap-3.5">' +

                '<div class="size-8 rounded-lg ' +
                cls.pill +
                ' flex items-center justify-center shrink-0">' +

                  '<span class="material-symbols-outlined text-lg">' +
                  "campaign" +
                  "</span>" +

                "</div>" +

                '<div class="flex-1 min-w-0">' +

                  '<div class="flex items-center justify-between flex-wrap gap-2 mb-1">' +

                    '<h4 class="text-sm font-bold text-white">' +
                    escapeHtml(title) +
                    "</h4>" +

                    '<span class="px-2 py-0.5 rounded text-[10px] font-extrabold uppercase font-mono tracking-wider ' +
                    cls.pill +
                    '">' +

                    escapeHtml(
                      level.toUpperCase()
                    ) +
                    " · LIVE" +

                    "</span>" +

                  "</div>" +

                  '<p class="text-xs text-slate-300 leading-relaxed">' +
                  escapeHtml(body) +
                  "</p>" +

                  '<div class="flex items-center gap-6 mt-3 text-[11px] text-slate-400 font-mono">' +

                    "<span>" +
                    "Valid: " +
                    '<strong class="text-white">' +
                    escapeHtml(validUntil) +
                    "</strong>" +
                    "</span>" +

                    "<span>" +
                    escapeHtml(issuer) +
                    "</span>" +

                  "</div>" +

                "</div>" +

              "</div>" +

            "</div>"
          );
        })
        .join("");
  }

  // ============================================================
  // BULLETIN GENERATOR
  // ============================================================

  function buildLiveBulletins(
    nowcastData,
    timelineData
  ) {
    if (
      !nowcastData ||
      !timelineData
    ) {
      return [];
    }

    var location =
      nowcastData.location ||
      "StormSense sector";

    var horizons =
      nowcastData.horizon_hazards ||
      {};

    var timeline =
      Array.isArray(
        timelineData.timeline
      )
        ? timelineData.timeline
        : [];

    function getProbability(
      hazard
    ) {
      return getHazardProbability(
        hazard
      );
    }

    function getLevel(
      hazard
    ) {
      if (!hazard) {
        return "green";
      }

      if (hazard.level) {
        return String(
          hazard.level
        ).toLowerCase();
      }

      return probabilityToLevel(
        getProbability(hazard)
      );
    }

    // ----------------------------------------------------------
    // THUNDERSTORM
    // ----------------------------------------------------------

    var thunderstorm30m =
      horizons["0.5"] &&
      horizons["0.5"].hazards
        ? (
            horizons["0.5"]
              .hazards
              .thunderstorm
          )
        : null;

    var thunderstorm6h =
      horizons["6"] &&
      horizons["6"].hazards
        ? (
            horizons["6"]
              .hazards
              .thunderstorm
          )
        : null;

    var thunderstorm30mPct =
      getProbability(
        thunderstorm30m
      );

    var thunderstorm6hPct =
      getProbability(
        thunderstorm6h
      );

    // ----------------------------------------------------------
    // RAINFALL
    // ----------------------------------------------------------

    var rainfall3h =
      null;

    var rainfall6h =
      null;

    timeline.forEach(
      function (point) {
        var hours =
          safeNumber(
            point.hours_from_now,
            0
          );

        if (
          hours > 2 &&
          hours <= 4
        ) {
          rainfall3h =
            point;
        }

        if (
          hours > 4 &&
          hours <= 7
        ) {
          rainfall6h =
            point;
        }
      }
    );

    var rainfall3hMm =
      rainfall3h
        ? safeNumber(
            rainfall3h.rainfall_mm,
            0
          )
        : 0;

    var rainfall6hMm =
      rainfall6h
        ? safeNumber(
            rainfall6h.rainfall_mm,
            0
          )
        : 0;

    // ----------------------------------------------------------
    // FLASH FLOOD
    // ----------------------------------------------------------

    var flashFlood30m =
      horizons["0.5"] &&
      horizons["0.5"].hazards
        ? (
            horizons["0.5"]
              .hazards
              .flash_flood
          )
        : null;

    var flashFlood6h =
      horizons["6"] &&
      horizons["6"].hazards
        ? (
            horizons["6"]
              .hazards
              .flash_flood
          )
        : null;

    var flashFlood30mPct =
      getProbability(
        flashFlood30m
      );

    var flashFlood6hPct =
      getProbability(
        flashFlood6h
      );

    // ----------------------------------------------------------
    // RETURN LIVE BULLETINS
    // ----------------------------------------------------------

    return [
      {
        title:
          "Thunderstorm Watch — " +
          location,

        level:
          getLevel(
            thunderstorm30m
          ),

        body:
          "Thunderstorm probability is " +
          thunderstorm30mPct +
          "% in the next 30 minutes, changing to " +
          thunderstorm6hPct +
          "% within 6 hours.",

        validUntil:
          "Next 6 hours",

        issuer:
          "StormSense Hazard Engine"
      },

      {
        title:
          "Rainfall Outlook — " +
          location,

        level:
          "green",

        body:
          "Forecast rainfall is " +
          rainfall3hMm.toFixed(2) +
          " mm around 3 hours, changing to " +
          rainfall6hMm.toFixed(2) +
          " mm around 6 hours.",

        validUntil:
          "Next 6 hours",

        issuer:
          "StormSense Hazard Engine"
      },

      {
        title:
          "Flash-Flood Risk — " +
          location,

        level:
          getLevel(
            flashFlood30m
          ),

        body:
          "Flash-flood probability is " +
          flashFlood30mPct +
          "% in the next 30 minutes and " +
          flashFlood6hPct +
          "% by 6 hours.",

        validUntil:
          "Next 6 hours",

        issuer:
          "StormSense Hazard Engine"
      }
    ];
  }

  // ============================================================
  // NOWCAST TIMELINE
  // ============================================================

  function renderNowcast(
    timeline
  ) {
    var root =
      document.getElementById(
        "nowcast-steps"
      );

    if (
      !root ||
      !Array.isArray(timeline) ||
      !timeline.length
    ) {
      return;
    }

    root.innerHTML =
      timeline
        .map(function (
          point,
          index
        ) {
          var label;

          if (index === 0) {
            label = "NOW";
          } else {
            label =
              "+" +
              safeNumber(
                point.hours_from_now,
                0
              ) +
              "h";
          }

          var rainfall =
            safeNumber(
              point.rainfall_mm,
              0
            ).toFixed(1);

          var temperature =
            safeNumber(
              point.temperature,
              0
            ).toFixed(1);

          var humidity =
            safeNumber(
              point.humidity,
              0
            ).toFixed(0);

          var wind =
            safeNumber(
              point.wind_speed,
              0
            ).toFixed(1);

          return (
            '<div class="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-center">' +

              '<div class="text-[10px] text-slate-400 font-mono uppercase">' +
              escapeHtml(label) +
              "</div>" +

              '<div class="text-white font-bold font-mono text-sm mt-1">' +
              temperature +
              "°C" +
              "</div>" +

              '<div class="text-[10px] text-blue-400 mt-1">' +
              rainfall +
              " mm" +
              "</div>" +

              '<div class="text-[10px] text-slate-400 mt-1">' +
              "Humidity " +
              humidity +
              "%" +
              "</div>" +

              '<div class="text-[10px] text-cyan-400 mt-1">' +
              "Wind " +
              wind +
              " km/h" +
              "</div>" +

              '<div class="text-[10px] text-slate-500 mt-1">' +
              escapeHtml(
                point.weather_description ||
                point.weather ||
                ""
              ) +
              "</div>" +

            "</div>"
          );
        })
        .join("");
  }

  // ============================================================
  // DISTRICT CARDS
  // ============================================================

  function renderDistricts(
    districts
  ) {
    var root =
      document.getElementById(
        "district-advisories-grid"
      );

    if (!root) {
      return;
    }

    if (
      !Array.isArray(districts) ||
      districts.length === 0
    ) {
      root.innerHTML =
        '<div class="p-4 text-sm text-slate-400">' +
        "Sector-level live risk is displayed above. District comparison data is not connected." +
        "</div>";

      return;
    }

    root.innerHTML =
      districts
        .map(function (d) {
          var cls =
            levelClass(
              d.riskLevel
            );

          var ring =
            d.isPrimary
              ? "border-2 " +
                cls.border
              : "border " +
                cls.border;

          return (
            '<div class="rounded-2xl ' +
            ring +
            ' bg-slate-900/80 p-6 shadow-xl">' +

              '<div class="flex items-start justify-between mb-3">' +

                "<div>" +

                  '<span class="font-mono text-xs font-extrabold uppercase tracking-wider ' +
                  cls.text +
                  '">' +

                  "REFERENCE · " +
                  escapeHtml(
                    String(
                      d.riskLevel ||
                      "green"
                    ).toUpperCase()
                  ) +

                  "</span>" +

                  '<h3 class="text-xl font-black text-white mt-1">' +
                  escapeHtml(
                    d.name ||
                    "Unknown District"
                  ) +
                  "</h3>" +

                "</div>" +

                '<span class="px-3 py-1 font-extrabold text-xs rounded-xl font-mono uppercase ' +
                cls.pill +
                '">' +

                escapeHtml(
                  d.riskLevel ||
                  "green"
                ) +

                "</span>" +

              "</div>" +

              '<p class="text-xs text-slate-300 mb-4">' +
              escapeHtml(
                d.note ||
                "Reference district information."
              ) +
              "</p>" +

              '<div class="grid grid-cols-3 gap-3 p-3 bg-slate-950/80 rounded-xl border border-slate-800 text-center font-mono">' +

                "<div>" +
                  '<span class="text-[10px] text-slate-400 block">TSTORM</span>' +
                  '<span class="' +
                  cls.text +
                  ' font-bold text-base">' +
                  safeNumber(
                    d.thunderstormPct,
                    0
                  ) +
                  "%" +
                  "</span>" +
                "</div>" +

                "<div>" +
                  '<span class="text-[10px] text-slate-400 block">HEAVY RAIN</span>' +
                  '<span class="' +
                  cls.text +
                  ' font-bold text-base">' +
                  safeNumber(
                    d.heavyRainfallPct,
                    0
                  ) +
                  "%" +
                  "</span>" +
                "</div>" +

                "<div>" +
                  '<span class="text-[10px] text-slate-400 block">FLASH FLOOD</span>' +
                  '<span class="' +
                  cls.text +
                  ' font-bold text-base">' +
                  safeNumber(
                    d.flashFloodPct,
                    0
                  ) +
                  "%" +
                  "</span>" +
                "</div>" +

              "</div>" +

            "</div>"
          );
        })
        .join("");
  }

  // ============================================================
  // XAI RENDERING
  // ============================================================

  function renderXai(
    xai
  ) {
    var inputs =
      xai &&
      xai.inputs
        ? xai.inputs
        : {};

    function contribution(
      group
    ) {
      if (
        xai &&
        xai.factors &&
        xai.factors[group] &&
        xai.factors[group].contribution != null
      ) {
        return xai.factors[group]
          .contribution;
      }

      return 0;
    }

    function renderRows(
      targetId
    ) {
      var root =
        document.getElementById(
          targetId
        );

      if (!root) {
        return;
      }

      if (
        !xai ||
        !xai.factors
      ) {
        root.innerHTML =
          '<div class="p-3 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-400 font-mono">' +
          "XAI data unavailable" +
          "</div>";

        return;
      }

      var factors = [
        {
          name:
            "Recent rainfall accumulation",

          value:
            safeNumber(
              inputs.rainfall_6h_mm,
              0
            ) +
            " mm / 6h",

          shapLabel:
            "+" +
            contribution(
              "rainfall"
            ) +
            "%"
        },

        {
          name:
            "Humidity / moisture",

          value:
            (
              inputs.humidity_percent != null
                ? inputs.humidity_percent
                : "—"
            ) +
            "%",

          shapLabel:
            "+" +
            contribution(
              "moisture"
            ) +
            "%"
        },

        {
          name:
            "Wind influence proxy",

          value:
            (
              inputs.wind_speed_kmh != null
                ? inputs.wind_speed_kmh
                : "—"
            ) +
            " km/h @ " +
            (
              inputs.wind_direction_deg != null
                ? inputs.wind_direction_deg
                : "—"
            ) +
            "°",

          shapLabel:
            "+" +
            contribution(
              "wind"
            ) +
            "%"
        },

        {
          name:
            "Instability proxy",

          value:
            "CAPE " +
            (
              inputs.cape_jkg != null
                ? inputs.cape_jkg
                : "—"
            ) +
            " J/kg",

          shapLabel:
            "+" +
            contribution(
              "instability"
            ) +
            "%"
        }
      ];

      if (
        xai.radar &&
        xai.radar.score != null &&
        xai.radar.contribution != null
      ) {
        factors.push({
          name:
            "Radar reflectivity proxy",

          value:
            (
              inputs.radar_dbz != null
                ? inputs.radar_dbz
                : "—"
            ) +
            " dBZ",

          shapLabel:
            "+" +
            xai.radar.contribution +
            "%"
        });
      }

      root.innerHTML =
        factors
          .map(function (f) {
            return (
              '<div class="p-3 rounded-xl bg-slate-950/70 border border-slate-800 flex items-center justify-between gap-3">' +

                "<div>" +

                  '<span class="text-xs font-bold text-white block">' +
                  escapeHtml(
                    f.name
                  ) +
                  "</span>" +

                  '<span class="text-[11px] text-slate-400 font-mono">' +
                  escapeHtml(
                    f.value
                  ) +
                  "</span>" +

                "</div>" +

                '<span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-purple-500/10 border border-purple-500/30 text-purple-300 whitespace-nowrap">' +
                escapeHtml(
                  f.shapLabel
                ) +
                "</span>" +

              "</div>"
            );
          })
          .join("");
    }

    renderRows(
      "xai-factors-dashboard"
    );

    renderRows(
      "xai-factors-lab"
    );

    setText(
      "xai-source",
      "LIVE · Open-Meteo · Rule-based atmospheric attribution"
    );
  }

  // ============================================================
  // RADAR HOTSPOTS
  // ============================================================

  function renderHotspots(
    hotspots
  ) {
    var root =
      document.getElementById(
        "radar-hotspots"
      );

    if (!root) {
      return;
    }

    if (
      !Array.isArray(hotspots)
    ) {
      root.innerHTML = "";
      return;
    }

    root.innerHTML =
      hotspots
        .map(function (h) {
          return (
            '<div class="absolute flex flex-col items-center" style="top:' +
            escapeHtml(h.top || "50%") +
            ";left:" +
            escapeHtml(h.left || "50%") +
            '">' +

              '<div class="size-12 rounded-full bg-amber-500/80 border-2 border-white flex flex-col items-center justify-center text-slate-950 shadow-lg font-mono text-[10px] font-black">' +
              "REFERENCE" +
              "</div>" +

              '<div class="mt-1 bg-slate-950/90 border border-amber-500/60 px-2 py-0.5 rounded text-[10px] font-bold text-amber-400">' +

              escapeHtml(
                h.name ||
                "Hotspot"
              ) +

              " · " +

              escapeHtml(
                h.level ||
                "WATCH"
              ) +

              "</div>" +

            "</div>"
          );
        })
        .join("");
  }

  // ============================================================
  // DASHBOARD MAP
  // ============================================================

  function getMapCoordinates(
    data
  ) {
    var lat =
      data &&
      data.location
        ? Number(
            data.location.lat
          )
        : DEFAULT_LOCATION.lat;

    var lon =
      data &&
      data.location
        ? Number(
            data.location.lon
          )
        : DEFAULT_LOCATION.lon;

    if (isNaN(lat)) {
      lat =
        DEFAULT_LOCATION.lat;
    }

    if (isNaN(lon)) {
      lon =
        DEFAULT_LOCATION.lon;
    }

    return {
      lat: lat,
      lon: lon
    };
  }

  function initMap(
    data
  ) {
    var mapContainer =
      document.getElementById(
        "map-radar-placeholder"
      );

    if (!mapContainer) {
      console.warn(
        "Dashboard map container not found."
      );

      return null;
    }

    if (
      typeof L === "undefined"
    ) {
      console.error(
        "Leaflet library is not loaded."
      );

      return null;
    }

    var coords =
      getMapCoordinates(
        data
      );

    if (
      window.stormSenseMap
    ) {
      window.stormSenseMap.setView(
        [
          coords.lat,
          coords.lon
        ],
        window.stormSenseMap.getZoom()
      );

      updateMapMarker(
        window.stormSenseMap,
        coords.lat,
        coords.lon
      );

      setTimeout(
        function () {
          window.stormSenseMap.invalidateSize();
        },
        100
      );

      return window.stormSenseMap;
    }

    var map =
      L.map(
        "map-radar-placeholder",
        {
          center: [
            coords.lat,
            coords.lon
          ],

          zoom: 10,

          minZoom: 6,

          maxZoom: 18,

          zoomControl: false
        }
      );

    L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',

    maxZoom: 19
  }
).addTo(map);

    var zoomInBtn =
      document.getElementById(
        "btn-map-zoom-in"
      );

    if (zoomInBtn) {
      zoomInBtn.addEventListener(
        "click",
        function (e) {
          e.preventDefault();
          e.stopPropagation();

          map.zoomIn();
        }
      );
    }

    var zoomOutBtn =
      document.getElementById(
        "btn-map-zoom-out"
      );

    if (zoomOutBtn) {
      zoomOutBtn.addEventListener(
        "click",
        function (e) {
          e.preventDefault();
          e.stopPropagation();

          map.zoomOut();
        }
      );
    }

    window.stormSenseMap =
      map;

    updateMapMarker(
      map,
      coords.lat,
      coords.lon
    );

    setTimeout(
      function () {
        map.invalidateSize();
      },
      200
    );

    return map;
  }

  // ============================================================
  // MAP MARKER
  // ============================================================

  function updateMapMarker(
    map,
    lat,
    lon
  ) {
    if (
      typeof L === "undefined" ||
      !map
    ) {
      return;
    }

    if (
      window.stormSenseSectorMarker
    ) {
      window.stormSenseSectorMarker.setLatLng(
        [
          lat,
          lon
        ]
      );

      return;
    }

    window.stormSenseSectorMarker =
      L.marker(
        [
          lat,
          lon
        ]
      )
        .addTo(map)
        .bindPopup(
          "StormSense live sector"
        );
  }

  // ============================================================
  // RADAR MAP
  // ============================================================

  function initRadarMap(
    data
  ) {
    var mapContainer =
      document.getElementById(
        "radar-map-container"
      );

    if (!mapContainer) {
      console.warn(
        "Radar map container not found."
      );

      return null;
    }

    if (
      typeof L === "undefined"
    ) {
      console.error(
        "Leaflet library is not loaded."
      );

      return null;
    }

    var coords =
      getMapCoordinates(
        data
      );

    if (
      window.stormSenseRadarMap
    ) {
      window.stormSenseRadarMap.setView(
        [
          coords.lat,
          coords.lon
        ],
        window.stormSenseRadarMap.getZoom()
      );

      setTimeout(
        function () {
          window.stormSenseRadarMap.invalidateSize();
        },
        100
      );

      return window.stormSenseRadarMap;
    }

    var map =
      L.map(
        "radar-map-container",
        {
          center: [
            coords.lat,
            coords.lon
          ],

          zoom: 10,

          minZoom: 6,

          maxZoom: 18,

          zoomControl: true
        }
      );

   L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',

    maxZoom: 19
  }
).addTo(map);

    window.stormSenseRadarMap =
      map;

      loadRainViewerRadar(
      map
    );

    setTimeout(
      function () {
        map.invalidateSize();
      },
      200
    );

    return map;
  }

  // ============================================================
  // UPDATE MAPS
  // ============================================================

  function updateMaps(
    data
  ) {
    if (
      !data ||
      !data.location
    ) {
      return;
    }

    var lat =
      Number(
        data.location.lat
      );

    var lon =
      Number(
        data.location.lon
      );

    if (
      isNaN(lat) ||
      isNaN(lon)
    ) {
      return;
    }

    if (
      window.stormSenseMap
    ) {
      window.stormSenseMap.setView(
        [
          lat,
          lon
        ],
        window.stormSenseMap.getZoom()
      );

      updateMapMarker(
        window.stormSenseMap,
        lat,
        lon
      );
    }

    if (
      window.stormSenseRadarMap
    ) {
      window.stormSenseRadarMap.setView(
        [
          lat,
          lon
        ],
        window.stormSenseRadarMap.getZoom()
      );
    }
  }

  // ============================================================
  // PAINT DASHBOARD
  // ============================================================

  function paintDashboard(
    data
  ) {
    if (!data) {
      console.error(
        "StormSense: no dashboard data."
      );

      return;
    }

    var loc =
      data.location || {};

    var cc =
      data.currentConditions || {};

    var hz =
      data.hazards || {};

    var st =
      data.systemStatus || {};

    var districts =
      Array.isArray(
        data.districts
      )
        ? data.districts
        : [];

    var primary =
      districts.filter(
        function (d) {
          return d.isPrimary;
        }
      )[0] ||
      districts[0];

    // ----------------------------------------------------------
    // LOCATION
    // ----------------------------------------------------------

    setText(
      "bind-district-name",
      loc.district ||
      "StormSense Sector"
    );

    setText(
      "bind-district-name-h2",
      (
        loc.district ||
        "StormSense Sector"
      ) +
      " Nowcasting Desk"
    );

    setText(
      "bind-location-line",
      loc.district ||
      ""
    );

    var lat =
      Number(loc.lat);

    var lon =
      Number(loc.lon);

    if (
      !isNaN(lat) &&
      !isNaN(lon)
    ) {
      setText(
        "bind-coords",
        lat.toFixed(2) +
        "°N, " +
        lon.toFixed(2) +
        "°E"
      );
    }

    // ----------------------------------------------------------
    // STATUS
    // ----------------------------------------------------------

    setText(
      "bind-timestamp",
      st.timestampDisplay ||
      ""
    );

    setText(
      "bind-freshness",
      st.freshnessLabel ||
      "LIVE · OpenWeather"
    );

    setText(
      "bind-disclaimer",
      data.disclaimer ||
      ""
    );

    setText(
      "bind-cluster",
      st.clusterLabel ||
      (
        "Sector: " +
        (
          data.sector ||
          currentSector
        )
      )
    );

    setText(
      "bind-sla",
      st.slaLabel ||
      ""
    );

    setText(
      "bind-version",
      st.versionLabel ||
      ""
    );

    setText(
      "bind-desk",
      st.deskLabel ||
      ""
    );

    // ----------------------------------------------------------
    // SIDEBAR
    // ----------------------------------------------------------

    setText(
      "bind-sidebar-alert-title",
      (
        loc.district ||
        "StormSense sector"
      ) +
      " hazard watch"
    );

    setText(
      "bind-sidebar-alert-body",
      "Live weather observations and StormSense rule-based hazard assessment."
    );

    // ----------------------------------------------------------
    // CURRENT WEATHER
    // ----------------------------------------------------------

    setText(
      "bind-temp",
      safeNumber(
        cc.temperatureC,
        0
      ).toFixed(1) +
      "°C"
    );

    setText(
      "bind-rain",
      safeNumber(
        cc.rainfallMmHr,
        0
      ).toFixed(1) +
      " mm/h"
    );

    setText(
      "bind-humidity",
      safeNumber(
        cc.humidityPct,
        0
      ).toFixed(0) +
      "%"
    );

    setText(
      "bind-wind",
      safeNumber(
        cc.windKmh,
        0
      ).toFixed(1) +
      " km/h " +
      (
        cc.windDirection ||
        ""
      )
    );

    // ----------------------------------------------------------
    // HAZARDS
    // ----------------------------------------------------------

    var thunderstorm =
      hz.thunderstorm ||
      adaptHazard(
        null,
        "Thunderstorm"
      );

    var heavyRainfall =
      hz.heavyRainfall ||
      adaptHazard(
        null,
        "Heavy Rainfall"
      );

    var flashFlood =
      hz.flashFlood ||
      adaptHazard(
        null,
        "Flash Flood"
      );

    var overall =
      hz.overall ||
      {
        probabilityPct: 0,
        level: "green",
        actionLabel:
          "NORMAL MONITORING",
        stageLabel:
          "NORMAL"
      };

    applyHazardCard(
      "kpi-ts",
      thunderstorm
    );

    applyHazardCard(
      "kpi-rain",
      heavyRainfall
    );

    applyHazardCard(
      "kpi-flood",
      flashFlood
    );

    setText(
      "kpi-overall-value",
      safeNumber(
        overall.probabilityPct,
        0
      ) +
      "%"
    );

    setText(
      "kpi-overall-level",
      String(
        overall.level ||
        "green"
      ).toUpperCase()
    );

    setText(
      "kpi-overall-action",
      overall.actionLabel ||
      getOverallAction(
        overall.level
      )
    );

    setText(
      "kpi-overall-stage",
      overall.stageLabel ||
      getOverallStage(
        overall.level
      )
    );

    setText(
      "kpi-overall-area",
      loc.district ||
      ""
    );

    setWidth(
      "kpi-overall-bar",
      overall.probabilityPct
    );

    // ----------------------------------------------------------
    // PROFILE
    // ----------------------------------------------------------

    setText(
      "profile-district",
      loc.district ||
      "StormSense Sector"
    );

    if (
      !isNaN(lat) &&
      !isNaN(lon)
    ) {
      setText(
        "profile-meta",
        "Lat " +
        lat.toFixed(2) +
        "°N, Lon " +
        lon.toFixed(2) +
        "°E · Live StormSense"
      );
    }

    setText(
      "profile-level",
      String(
        primary &&
        primary.riskLevel
          ? primary.riskLevel
          : overall.level ||
            "green"
      ).toUpperCase()
    );

    // ----------------------------------------------------------
    // RISK METERS
    // ----------------------------------------------------------

    setText(
      "meter-ts-label",
      thunderstorm.probabilityPct +
      "% (" +
      thunderstorm.level +
      ")"
    );

    setText(
      "meter-rain-label",
      heavyRainfall.probabilityPct +
      "% (" +
      heavyRainfall.level +
      ")"
    );

    setText(
      "meter-flood-label",
      flashFlood.probabilityPct +
      "% (" +
      flashFlood.level +
      ")"
    );

    setWidth(
      "meter-ts-bar",
      thunderstorm.probabilityPct
    );

    setWidth(
      "meter-rain-bar",
      heavyRainfall.probabilityPct
    );

    setWidth(
      "meter-flood-bar",
      flashFlood.probabilityPct
    );

    // ----------------------------------------------------------
    // SELECTOR
    // ----------------------------------------------------------

    var sectorSelect =
      document.getElementById(
        "sector-selector"
      );

    if (sectorSelect) {
      sectorSelect.value =
        data.sector ||
        currentSector ||
        "default";
    }

    // ----------------------------------------------------------
    // COMPONENTS
    // ----------------------------------------------------------

    renderBulletins(
      buildLiveBulletins(
        window.StormSenseNowcastData,
        window.StormSenseLiveData
      )
    );

    renderNowcast(
      data.timeline || []
    );

    renderDistricts(
      districts
    );

    renderXai(
      data.xai
    );

    if (
      data.radarPlaceholder
    ) {
      renderHotspots(
        data.radarPlaceholder.hotspots
      );
    }

    setText(
      "nav-advisory-count",
      "LIVE"
    );

    console.log(
      "StormSense dashboard painted:",
      data.sector,
      data.location
    );
  }

  // ============================================================
  // SECTOR REFRESH
  // ============================================================

  function refreshSector(
    sector
  ) {
    var selectedSector =
      sector || "default";

    currentSector =
      selectedSector;

    if (
      typeof window.showToast ===
      "function"
    ) {
      window.showToast(
        "Loading live " +
        selectedSector +
        " data...",
        "radar"
      );
    }

    var sectorSelect =
      document.getElementById(
        "sector-selector"
      );

    if (sectorSelect) {
      sectorSelect.disabled =
        true;
    }

    return loadDashboardData(
      selectedSector
    )
      .then(function (data) {
        if (!data) {
          throw new Error(
            "No dashboard data returned."
          );
        }

        window.StormSenseLiveDashboardData =
          data;

        paintDashboard(
          data
        );

        updateMaps(
          data
        );

        if (
          typeof window.showToast ===
          "function"
        ) {
          window.showToast(
            (
              data.location &&
              data.location.district
            ) ||
            data.location ||
            selectedSector +
              " · LIVE data loaded",
            "check_circle"
          );
        }

        return data;
      })
      .catch(function (error) {
        console.error(
          "Sector refresh failed:",
          error
        );

        if (
          typeof window.showToast ===
          "function"
        ) {
          window.showToast(
            "Unable to load live sector data.",
            "warning"
          );
        }

        throw error;
      })
      .finally(function () {
        if (sectorSelect) {
          sectorSelect.disabled =
            false;
        }
      });
  }

  // ============================================================
  // VIEW SWITCHING
  // ============================================================

  window.switchNowcastView =
    function (
      targetView
    ) {
      viewIds.forEach(
        function (view) {
          var container =
            document.getElementById(
              "view-" +
              view
            );

          if (!container) {
            return;
          }

          if (
            view ===
            targetView
          ) {
            container.classList.remove(
              "hidden"
            );
          } else {
            container.classList.add(
              "hidden"
            );
          }
        }
      );

      document
        .querySelectorAll(
          ".nav-item"
        )
        .forEach(
          function (item) {
            var view =
              item.getAttribute(
                "data-view"
              );

            item.classList.remove(
              "bg-slate-800/80",
              "text-white",
              "font-semibold",
              "border-cyan-500/40",
              "shadow-lg",
              "shadow-cyan-500/10",
              "border-slate-700/60"
            );

            if (
              view ===
              targetView
            ) {
              item.classList.remove(
                "text-slate-300",
                "font-medium"
              );

              item.classList.add(
                "bg-slate-800/80",
                "text-white",
                "font-semibold",
                "border-slate-700/60"
              );
            } else {
              item.classList.add(
                "text-slate-300",
                "font-medium"
              );
            }
          }
        );

      if (
        targetView ===
          "dashboard" &&
        window.stormSenseMap
      ) {
        setTimeout(
          function () {
            window.stormSenseMap.invalidateSize();
          },
          150
        );
      }

      if (
        targetView ===
        "radar"
      ) {
        var data =
          window.StormSenseLiveDashboardData;

        if (
          data &&
          !window.stormSenseRadarMap
        ) {
          initRadarMap(
            data
          );
        } else if (
          window.stormSenseRadarMap
        ) {
          setTimeout(
            function () {
              window.stormSenseRadarMap.invalidateSize();
            },
            150
          );
        }
      }

      window.scrollTo({
        top: 0,
        behavior: "smooth"
      });
    };

  // ============================================================
  // FORECAST HORIZON
  // ============================================================

  window.setForecastHorizon =
    function (
      btn,
      label
    ) {
      document
        .querySelectorAll(
          ".horizon-btn"
        )
        .forEach(
          function (b) {
            b.classList.remove(
              "bg-red-600",
              "text-white",
              "font-bold"
            );

            b.classList.add(
              "text-slate-400",
              "font-medium"
            );
          }
        );

      if (btn) {
        btn.classList.add(
          "bg-red-600",
          "text-white",
          "font-bold"
        );

        btn.classList.remove(
          "text-slate-400",
          "font-medium"
        );
      }

      var timelineData =
        window.StormSenseLiveData;

      var nowcastData =
        window.StormSenseNowcastData;

      if (
        !timelineData ||
        !timelineData.timeline ||
        !timelineData.timeline.length
      ) {
        window.showToast(
          "Live forecast data unavailable",
          "warning"
        );

        return;
      }

      var targetHours =
        0;

      if (
        label === "+30m"
      ) {
        targetHours =
          0.5;
      } else if (
        label === "+2 Hours"
      ) {
        targetHours =
          2;
      } else if (
        label === "+4 Hours"
      ) {
        targetHours =
          4;
      } else if (
        label === "+6 Hours"
      ) {
        targetHours =
          6;
      }

      var selectedPoint =
        timelineData.timeline[0];

      var smallestDifference =
        Infinity;

      timelineData.timeline.forEach(
        function (point) {
          var pointHours =
            safeNumber(
              point.hours_from_now,
              0
            );

          var difference =
            Math.abs(
              pointHours -
              targetHours
            );

          if (
            difference <
            smallestDifference
          ) {
            smallestDifference =
              difference;

            selectedPoint =
              point;
          }
        }
      );

      // --------------------------------------------------------
      // WEATHER FOR SELECTED HORIZON
      // --------------------------------------------------------

      var temperature =
        safeNumber(
          selectedPoint.temperature,
          0
        );

      var rainfall =
        safeNumber(
          selectedPoint.rainfall_mm,
          0
        );

      var humidity =
        safeNumber(
          selectedPoint.humidity,
          0
        );

      var wind =
        safeNumber(
          selectedPoint.wind_speed,
          0
        );

      var windDirection =
        getWindDirection(
          selectedPoint.wind_direction
        );

      setText(
        "bind-temp",
        temperature.toFixed(1) +
        "°C"
      );

      setText(
        "bind-rain",
        rainfall.toFixed(1) +
        " mm/h"
      );

      setText(
        "bind-humidity",
        humidity.toFixed(0) +
        "%"
      );

      setText(
        "bind-wind",
        wind.toFixed(1) +
        " km/h " +
        windDirection
      );

      // --------------------------------------------------------
      // SOURCE LABEL
      // --------------------------------------------------------

      var actualHours =
        safeNumber(
          selectedPoint.hours_from_now,
          0
        );

      var sourceText;

      if (
        actualHours === 0
      ) {
        sourceText =
          "Live observation · OpenWeather";
      } else {
        sourceText =
          "Forecast +" +
          actualHours.toFixed(1) +
          "h · OpenWeather";
      }

      var sourceLabels =
        document.querySelectorAll(
          "#bind-temp + span, " +
          "#bind-rain + span, " +
          "#bind-humidity + span, " +
          "#bind-wind + span"
        );

      sourceLabels.forEach(
        function (el) {
          el.textContent =
            sourceText;
        }
      );

      // --------------------------------------------------------
      // HORIZON HAZARDS
      // --------------------------------------------------------

      var horizonHazards =
        null;

      if (
        nowcastData &&
        nowcastData.horizon_hazards
      ) {
        horizonHazards =
          nowcastData.horizon_hazards[
            String(
              targetHours
            )
          ];
      }

      if (
        horizonHazards &&
        horizonHazards.hazards
      ) {
        var horizon =
          horizonHazards.hazards;

        var thunderstorm =
          adaptHazard(
            horizon.thunderstorm,
            "Thunderstorm"
          );

        var heavyRainfall =
          adaptHazard(
            horizon.heavy_rainfall ||
            horizon.heavyRainfall,
            "Heavy Rainfall"
          );

        var flashFlood =
          adaptHazard(
            horizon.flash_flood ||
            horizon.flashFlood,
            "Flash Flood"
          );

        applyHazardCard(
          "kpi-ts",
          thunderstorm
        );

        applyHazardCard(
          "kpi-rain",
          heavyRainfall
        );

        applyHazardCard(
          "kpi-flood",
          flashFlood
        );

        var overall =
          horizon.overall ||
          {};

        var overallProbability =
          getHazardProbability(
            overall
          );

        var overallLevel =
          overall.level ||
          probabilityToLevel(
            overallProbability
          );

        setText(
          "kpi-overall-value",
          overallProbability +
          "%"
        );

        setText(
          "kpi-overall-level",
          String(
            overallLevel
          ).toUpperCase()
        );

        setText(
          "kpi-overall-action",
          getOverallAction(
            overallLevel
          )
        );

        setText(
          "kpi-overall-stage",
          getOverallStage(
            overallLevel
          )
        );

        setWidth(
          "kpi-overall-bar",
          overallProbability
        );

        setText(
          "profile-level",
          String(
            overallLevel
          ).toUpperCase()
        );

        // ------------------------------------------------------
        // RISK METERS
        // ------------------------------------------------------

        setText(
          "meter-ts-label",
          getHazardProbability(
            horizon.thunderstorm
          ) +
          "% (" +
          (
            horizon.thunderstorm &&
            horizon.thunderstorm.level
              ? horizon.thunderstorm.level
              : probabilityToLevel(
                  getHazardProbability(
                    horizon.thunderstorm
                  )
                )
          ) +
          ")"
        );

        setText(
          "meter-rain-label",
          getHazardProbability(
            horizon.heavy_rainfall ||
            horizon.heavyRainfall
          ) +
          "% (" +
          (
            (
              horizon.heavy_rainfall ||
              horizon.heavyRainfall
            ) &&
            (
              horizon.heavy_rainfall ||
              horizon.heavyRainfall
            ).level
              ? (
                  horizon.heavy_rainfall ||
                  horizon.heavyRainfall
                ).level
              : probabilityToLevel(
                  getHazardProbability(
                    horizon.heavy_rainfall ||
                    horizon.heavyRainfall
                  )
                )
          ) +
          ")"
        );

        setText(
          "meter-flood-label",
          getHazardProbability(
            horizon.flash_flood ||
            horizon.flashFlood
          ) +
          "% (" +
          (
            (
              horizon.flash_flood ||
              horizon.flashFlood
            ) &&
            (
              horizon.flash_flood ||
              horizon.flashFlood
            ).level
              ? (
                  horizon.flash_flood ||
                  horizon.flashFlood
                ).level
              : probabilityToLevel(
                  getHazardProbability(
                    horizon.flash_flood ||
                    horizon.flashFlood
                  )
                )
          ) +
          ")"
        );

        setWidth(
          "meter-ts-bar",
          getHazardProbability(
            horizon.thunderstorm
          )
        );

        setWidth(
          "meter-rain-bar",
          getHazardProbability(
            horizon.heavy_rainfall ||
            horizon.heavyRainfall
          )
        );

        setWidth(
          "meter-flood-bar",
          getHazardProbability(
            horizon.flash_flood ||
            horizon.flashFlood
          )
        );

        console.log(
          "StormSense horizon hazards:",
          targetHours,
          horizonHazards
        );
      } else {
        console.warn(
          "StormSense: no horizon hazard data found for",
          targetHours
        );
      }

      window.showToast(
        label +
        " → using +" +
        actualHours.toFixed(1) +
        "h OpenWeather data",
        "radar"
      );
    };

  // ============================================================
  // SIREN
  // ============================================================

  window.triggerSiren =
    function () {
      var modal =
        document.getElementById(
          "modal-siren"
        );

      if (modal) {
        modal.classList.remove(
          "hidden"
        );
      }
    };

  window.closeSirenModal =
    function () {
      var modal =
        document.getElementById(
          "modal-siren"
        );

      if (modal) {
        modal.classList.add(
          "hidden"
        );
      }
    };

  window.executeSirenDispatch =
    function () {
      window.closeSirenModal();

      window.showToast(
        "DEMO ONLY — no siren was dispatched.",
        "warning"
      );
    };

  // ============================================================
  // TOAST
  // ============================================================

  window.showToast =
    function (
      msg,
      iconType
    ) {
      var toast =
        document.getElementById(
          "toast-notification"
        );

      var text =
        document.getElementById(
          "toast-message"
        );

      var icon =
        document.getElementById(
          "toast-icon"
        );

      if (
        !toast ||
        !text
      ) {
        console.log(
          "StormSense toast:",
          msg
        );

        return;
      }

      text.textContent =
        msg;

      if (icon) {
        if (
          iconType ===
          "warning"
        ) {
          icon.textContent =
            "warning";

          icon.className =
            "material-symbols-outlined text-red-400";
        } else if (
          iconType ===
          "radar"
        ) {
          icon.textContent =
            "radar";

          icon.className =
            "material-symbols-outlined text-blue-400";
        } else {
          icon.textContent =
            "check_circle";

          icon.className =
            "material-symbols-outlined text-emerald-400";
        }
      }

      toast.classList.remove(
        "translate-y-20",
        "opacity-0"
      );

      toast.classList.add(
        "translate-y-0",
        "opacity-100"
      );

      setTimeout(
        function () {
          toast.classList.remove(
            "translate-y-0",
            "opacity-100"
          );

          toast.classList.add(
            "translate-y-20",
            "opacity-0"
          );
        },
        4000
      );
    };

  // ============================================================
  // USER PINPOINT CARD
  // ============================================================

  window.toggleUserPinpointCard =
    function () {
      var card =
        document.getElementById(
          "user-pinpoint-card"
        );

      if (!card) {
        return;
      }

      if (
        card.classList.contains(
          "hidden"
        )
      ) {
        card.classList.remove(
          "hidden"
        );

        card.classList.add(
          "flex"
        );
      } else {
        card.classList.add(
          "hidden"
        );

        card.classList.remove(
          "flex"
        );
      }
    };

  // ============================================================
  // RECENTER
  // ============================================================

  window.recenterOnUser =
    function () {
      if (
        window.stormSenseMap &&
        window.StormSenseLiveDashboardData &&
        window.StormSenseLiveDashboardData.location
      ) {
        var lat =
          Number(
            window.StormSenseLiveDashboardData
              .location
              .lat
          );

        var lon =
          Number(
            window.StormSenseLiveDashboardData
              .location
              .lon
          );

        if (
          !isNaN(lat) &&
          !isNaN(lon)
        ) {
          window.stormSenseMap.setView(
            [
              lat,
              lon
            ],
            11
          );
        }
      }

      window.showToast(
        "Map centered on active StormSense sector.",
        "check_circle"
      );
    };

  // ============================================================
  // BROWSER GEOLOCATION
  // ============================================================

  window.handleMyLocationClick =
    function () {
      var ping =
        document.getElementById(
          "my-location-ping"
        );

      var marker =
        document.getElementById(
          "user-location-marker"
        );

      var coordsReadout =
        document.getElementById(
          "user-marker-coords"
        );

      var card =
        document.getElementById(
          "user-pinpoint-card"
        );

      if (ping) {
        ping.classList.remove(
          "hidden"
        );
      }

      window.showToast(
        "Requesting browser location...",
        "radar"
      );

      function applyLocation(
        lat,
        lon,
        label
      ) {
        if (marker) {
          marker.classList.remove(
            "hidden"
          );

          marker.classList.add(
            "flex"
          );

          marker.style.top =
            "48%";

          marker.style.left =
            "49%";
        }

        if (coordsReadout) {
          coordsReadout.textContent =
            Number(lat).toFixed(3) +
            "°N, " +
            Number(lon).toFixed(3) +
            "°E";
        }

        setText(
          "user-pinpoint-name",
          label
        );

        if (card) {
          card.classList.remove(
            "hidden"
          );

          card.classList.add(
            "flex"
          );
        }

        window.showToast(
          "GPS overlay: " +
          label,
          "check_circle"
        );
      }

      if (
        "geolocation" in
        navigator
      ) {
        navigator.geolocation.getCurrentPosition(
          function (pos) {
            applyLocation(
              pos.coords.latitude,
              pos.coords.longitude,
              "Browser GPS overlay"
            );
          },

          function () {
            var data =
              window.StormSenseLiveDashboardData;

            var lat =
              data &&
              data.location
                ? data.location.lat
                : DEFAULT_LOCATION.lat;

            var lon =
              data &&
              data.location
                ? data.location.lon
                : DEFAULT_LOCATION.lon;

            applyLocation(
              lat,
              lon,
              "Active StormSense sector"
            );
          },

          {
            timeout: 3500,
            enableHighAccuracy: true
          }
        );
      } else {
        window.showToast(
          "Browser geolocation is unavailable.",
          "warning"
        );
      }
    };

  // ============================================================
  // DEMO STUB
  // ============================================================

  window.stubDemoAction =
    function (
      label
    ) {
      window.showToast(
        "DEMO stub: " +
        label +
        " — not connected.",
        "warning"
      );
    };

  // ============================================================
  // DOM READY
  // ============================================================

  document.addEventListener(
    "DOMContentLoaded",
    function () {
      console.log(
        "StormSense frontend initializing..."
      );

      // --------------------------------------------------------
      // NAVIGATION
      // --------------------------------------------------------

      document
        .querySelectorAll(
          ".nav-item"
        )
        .forEach(
          function (btn) {
            btn.addEventListener(
              "click",
              function (e) {
                e.preventDefault();

                var view =
                  this.getAttribute(
                    "data-view"
                  );

                if (view) {
                  window.switchNowcastView(
                    view
                  );
                }
              }
            );
          }
        );

      // --------------------------------------------------------
      // SIREN
      // --------------------------------------------------------

      document
        .querySelectorAll(
          ".btn-siren-trigger"
        )
        .forEach(
          function (btn) {
            btn.addEventListener(
              "click",
              function (e) {
                e.preventDefault();

                window.triggerSiren();
              }
            );
          }
        );

      // --------------------------------------------------------
      // EXPORT
      // --------------------------------------------------------

      var exportBtn =
        document.getElementById(
          "btn-export-report"
        );

      if (exportBtn) {
        exportBtn.addEventListener(
          "click",
          function (e) {
            e.preventDefault();

            window.showToast(
              "DEMO — no GRIB/FITS export.",
              "check_circle"
            );
          }
        );
      }

      // --------------------------------------------------------
      // SECTOR SELECTOR
      // --------------------------------------------------------

      var sectorSelect =
        document.getElementById(
          "sector-selector"
        );

      if (sectorSelect) {
        sectorSelect.addEventListener(
          "change",
          function () {
            var selectedSector =
              this.value ||
              "default";

            if (
              selectedSector ===
              currentSector
            ) {
              return;
            }

            refreshSector(
              selectedSector
            ).catch(
              function () {
                // Error already handled by refreshSector().
              }
            );
          }
        );
      } else {
        console.warn(
          "StormSense: #sector-selector not found."
        );
      }

      // --------------------------------------------------------
      // INITIAL LIVE LOAD
      // --------------------------------------------------------

      console.log(
        "StormSense: starting initial live API load..."
      );

      loadDashboardData(
        currentSector
      )
        .then(
          function (data) {
            if (!data) {
              throw new Error(
                "StormSense data missing."
              );
            }

            console.log(
              "StormSense live data loaded:",
              data
            );

            window.StormSenseLiveDashboardData =
              data;

            paintDashboard(
              data
            );

            initMap(
              data
            );

            initRadarMap(
              data
            );

            window.switchNowcastView(
              "dashboard"
            );

            console.log(
              "StormSense frontend initialized successfully."
            );
          }
        )
        .catch(
          function (error) {
            console.error(
              "StormSense initialization failed:",
              error
            );

            window.showToast(
              "Live StormSense backend unavailable. Check browser console.",
              "warning"
            );
          }
        );
    }
  );

})();