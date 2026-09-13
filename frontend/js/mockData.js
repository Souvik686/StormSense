/**
 * StormSense Geographic Reference Configuration
 * Static administrative metadata for the monitored North 24 Parganas sector.
 * All weather values and model predictions are loaded dynamically from authentic APIs.
 */
(function (global) {
  "use strict";

  global.StormSenseConfig = {
    location: {
      district: "North 24 Parganas",
      state: "West Bengal",
      country: "India",
      shortLabel: "North 24 Parganas, West Bengal",
      lat: 22.724,
      lon: 88.479,
      subdivisions: ["Barasat Sadar", "Barrackpore", "Bangaon", "Basirhat", "Bidhannagar"],
      defaultSector: "Barasat Sadar"
    }
  };

  // Backward-compatible reference without fabricated data
  global.StormSenseMockData = global.StormSenseConfig;
})(window);
