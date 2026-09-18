// ── Google Maps Leaflet Compatibility Shim ──────────────────────────────────
//
// The StormSense frontend was written against the Leaflet API. The visible base
// map is now Google Maps 2D, so this layer translates the Leaflet calls the app
// actually makes into Google Maps operations.
//
// Scope rule: every method below is backed by a real Google Maps operation
// wherever Google has an equivalent. Methods with no Google counterpart
// (Leaflet pane z-ordering) are deliberate no-ops and are marked as such --
// they are the only no-ops in this file.
//
// StormSense products (risk surface, radar mosaic, boundaries, markers) stay
// separate from the base map: they are drawn as GroundOverlays, Data-layer
// features, tile overlays and Markers on top of it.
(function () {
  'use strict';

  var DARK_STYLE = [
    {elementType: 'geometry', stylers: [{color: '#242f3e'}]},
    {elementType: 'labels.text.stroke', stylers: [{color: '#242f3e'}]},
    {elementType: 'labels.text.fill', stylers: [{color: '#746855'}]},
    {featureType: 'administrative.locality', elementType: 'labels.text.fill', stylers: [{color: '#d59563'}]},
    {featureType: 'poi', elementType: 'labels.text.fill', stylers: [{color: '#d59563'}]},
    {featureType: 'poi.park', elementType: 'geometry', stylers: [{color: '#263c3f'}]},
    {featureType: 'poi.park', elementType: 'labels.text.fill', stylers: [{color: '#6b9a76'}]},
    {featureType: 'road', elementType: 'geometry', stylers: [{color: '#38414e'}]},
    {featureType: 'road', elementType: 'geometry.stroke', stylers: [{color: '#212a37'}]},
    {featureType: 'road', elementType: 'labels.text.fill', stylers: [{color: '#9ca5b3'}]},
    {featureType: 'road.highway', elementType: 'geometry', stylers: [{color: '#746855'}]},
    {featureType: 'road.highway', elementType: 'geometry.stroke', stylers: [{color: '#1f2835'}]},
    {featureType: 'road.highway', elementType: 'labels.text.fill', stylers: [{color: '#f3d19c'}]},
    {featureType: 'transit', elementType: 'geometry', stylers: [{color: '#2f3948'}]},
    {featureType: 'transit.station', elementType: 'labels.text.fill', stylers: [{color: '#d59563'}]},
    {featureType: 'water', elementType: 'geometry', stylers: [{color: '#17263c'}]},
    {featureType: 'water', elementType: 'labels.text.fill', stylers: [{color: '#515c6d'}]},
    {featureType: 'water', elementType: 'labels.text.stroke', stylers: [{color: '#17263c'}]}
  ];

  // Leaflet accepts [lat, lng], {lat, lng} and LatLng. Normalise to a Google
  // LatLngLiteral so every call site keeps working unchanged.
  function toLatLng(v) {
    if (v == null) return null;
    if (Array.isArray(v)) return {lat: Number(v[0]), lng: Number(v[1])};
    if (typeof v.lat === 'function') return {lat: v.lat(), lng: v.lng()};
    if (typeof v.lat === 'number') {
      return {lat: v.lat, lng: (v.lng != null ? v.lng : v.lon)};
    }
    return null;
  }

  // Accepts a shim bounds object, a raw google.maps.LatLngBounds, or Leaflet
  // [[s,w],[n,e]] corner arrays (passed directly at several fitBounds sites).
  function toGBounds(b) {
    if (!b) return null;
    if (b._gbounds) return b._gbounds;
    if (typeof b.getSouthWest === 'function') return b;
    if (Array.isArray(b) && b.length === 2) {
      var a = toLatLng(b[0]), c = toLatLng(b[1]);
      if (!a || !c) return null;
      return new google.maps.LatLngBounds(
        {lat: Math.min(a.lat, c.lat), lng: Math.min(a.lng, c.lng)},
        {lat: Math.max(a.lat, c.lat), lng: Math.max(a.lng, c.lng)}
      );
    }
    return null;
  }

  // Leaflet padding is [x, y] pixels; Google takes {top,right,bottom,left}.
  function toPadding(opts) {
    if (!opts || !opts.padding) return undefined;
    var p = opts.padding;
    var x = Array.isArray(p) ? Number(p[0]) : Number(p);
    var y = Array.isArray(p) ? Number(p[1]) : Number(p);
    return {top: y, bottom: y, left: x, right: x};
  }

  function wrapBounds(gb) {
    return {
      _gbounds: gb,
      getSouthWest: function () { return gb.getSouthWest(); },
      getNorthEast: function () { return gb.getNorthEast(); },
      getCenter: function () { return gb.getCenter(); },
      contains: function (ll) { return gb.contains(toLatLng(ll)); },
      isValid: function () { return !gb.isEmpty(); },
      // Leaflet pad(f) expands the box by a fraction of its own size. initMap()
      // builds the pan constraint from it, so this must really expand --
      // returning the same box would lock the map to the tight frame.
      pad: function (f) {
        var sw = gb.getSouthWest(), ne = gb.getNorthEast();
        var dLat = (ne.lat() - sw.lat()) * f;
        var dLng = (ne.lng() - sw.lng()) * f;
        return wrapBounds(new google.maps.LatLngBounds(
          {lat: sw.lat() - dLat, lng: sw.lng() - dLng},
          {lat: ne.lat() + dLat, lng: ne.lng() + dLng}
        ));
      }
    };
  }

  // Leaflet style keys -> Google Maps style keys. Shared by the Data layer and
  // by per-feature setStyle() on hover.
  function toGStyle(lStyle) {
    lStyle = lStyle || {};
    var transparentFill = (lStyle.fillColor === 'transparent');
    var g = {
      strokeColor: lStyle.color,
      strokeWeight: lStyle.weight,
      strokeOpacity: lStyle.opacity != null ? lStyle.opacity : 1,
      fillColor: transparentFill ? '#000000' : lStyle.fillColor,
      fillOpacity: transparentFill ? 0 : lStyle.fillOpacity,
      clickable: lStyle.interactive !== false
    };
    Object.keys(g).forEach(function (k) { if (g[k] === undefined) delete g[k]; });
    return g;
  }

  window.L = {
    // ── Map ──────────────────────────────────────────────────────────────────
    map: function (id, options) {
      options = options || {};
      var el = typeof id === 'string' ? document.getElementById(id) : id;

      // Google Maps CLEARS its container's children when it takes it over;
      // Leaflet left them alone. The dashboard authors its map HUD directly
      // inside #map-radar-placeholder -- the risk legend (#map-legend-title,
      // #map-legend-pills, #map-legend-subtext, #legend-lead-tag,
      // #obs-provenance), the domain toggle, the my-location button and the
      // zoom buttons. Constructing the map therefore deleted all of them:
      // applyHorizonLegend() then wrote to elements that no longer existed, so
      // the legend never tracked the selected horizon and the HUD buttons were
      // gone. Detach them first, let Google build its own DOM, then re-attach
      // them above the map. They are absolutely positioned already, so they
      // return to their intended places.
      var hud = [];
      if (el) {
        for (var ci = 0; ci < el.children.length; ci++) hud.push(el.children[ci]);
      }

      var gmap = new google.maps.Map(el, {
        center: toLatLng(options.center) || {lat: 23.8, lng: 87.9},
        zoom: options.zoom,
        minZoom: options.minZoom,
        maxZoom: options.maxZoom,
        zoomControl: !!options.zoomControl,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false,
        rotateControl: false,
        scaleControl: false,
        disableDefaultUI: !options.zoomControl,
        // Navigation must work everywhere on the canvas, including over the
        // StormSense risk surface. 'greedy' means the wheel zooms without
        // requiring Ctrl, and these three are set EXPLICITLY rather than left
        // to Google's defaults so a future options change cannot silently
        // disable panning or wheel zoom over the overlay.
        gestureHandling: 'greedy',
        draggable: true,
        scrollwheel: true,
        disableDoubleClickZoom: false,
        keyboardShortcuts: true,
        styles: DARK_STYLE,
        backgroundColor: '#0B0F19'
      });

      var lMap = {
        _gmap: gmap,
        _container: el,
        _layers: [],
        _openInfoWindows: [],

        // isRadarFeedsMap() gates RainViewer on the container id, so this has
        // to return the real element, not a stub.
        getContainer: function () { return el; },

        on: function (event, handler) {
          if (event === 'click') {
            gmap.addListener('click', function (e) {
              handler({latlng: {lat: e.latLng.lat(), lng: e.latLng.lng()}});
            });
          } else if (event === 'zoomend') {
            gmap.addListener('zoom_changed', function () { handler({}); });
          } else if (event === 'moveend') {
            gmap.addListener('idle', function () { handler({}); });
          }
          return this;
        },
        addListener: function (ev, fn) { return gmap.addListener(ev, fn); },

        setView: function (center, zoom) {
          gmap.setCenter(toLatLng(center));
          if (zoom != null) gmap.setZoom(zoom);
          return this;
        },
        panTo: function (center) { gmap.panTo(toLatLng(center)); return this; },

        // Area selection depends on this. Google panTo animates, giving the
        // Leaflet flyTo behaviour the call sites expect.
        // Leaflet flyTo is an ANIMATED but ATOMIC move: the target centre and
        // zoom are reached together.
        //
        // Mapping it to panTo() + setZoom() was subtly wrong and caused the
        // "Live Location needs two clicks" bug. panTo() animates; the setZoom()
        // issued on the very next line interrupted that animation, and while
        // the map was still mid-flight Google's `restriction` box re-clamped
        // the centre. The first click therefore landed somewhere between the
        // old and new positions, and only a second click -- starting from
        // closer in, with a shorter pan -- arrived at the real coordinates.
        //
        // Setting zoom FIRST and then centring instantaneously makes the move
        // deterministic: one call, one final position, no animation for a
        // later call to interrupt. Panning by hand is unaffected.
        flyTo: function (center, zoom) {
          if (zoom != null) gmap.setZoom(zoom);
          gmap.setCenter(toLatLng(center));
          return this;
        },

        setCenter: function (c) { gmap.setCenter(toLatLng(c)); return this; },
        getCenter: function () { return gmap.getCenter(); },
        setZoom: function (z) { gmap.setZoom(z); return this; },
        getZoom: function () { return gmap.getZoom(); },
        zoomIn: function () { gmap.setZoom(gmap.getZoom() + 1); return this; },
        zoomOut: function () { gmap.setZoom(gmap.getZoom() - 1); return this; },
        setMinZoom: function (z) { gmap.setOptions({minZoom: z}); return this; },
        setMaxZoom: function (z) { gmap.setOptions({maxZoom: z}); return this; },

        // Leaflet constrains panning to the box; Google calls it restriction.
        setMaxBounds: function (b) {
          var gb = toGBounds(b);
          if (gb) gmap.setOptions({restriction: {latLngBounds: gb, strictBounds: false}});
          return this;
        },

        fitBounds: function (bounds, opts) {
          var gb = toGBounds(bounds);
          if (gb) gmap.fitBounds(gb, toPadding(opts));
          return this;
        },

        // Leaflet returns the zoom at which a box fits the viewport. Google has
        // no direct equivalent, so derive it from the box span and the
        // container size using the same Web Mercator tile math.
        getBoundsZoom: function (bounds) {
          var gb = toGBounds(bounds);
          if (!gb) return gmap.getZoom();
          var sw = gb.getSouthWest(), ne = gb.getNorthEast();
          var w = (el && el.clientWidth) || 800;
          var h = (el && el.clientHeight) || 600;
          function latRad(lat) {
            var s = Math.sin(lat * Math.PI / 180);
            return Math.log((1 + s) / (1 - s)) / 2;
          }
          var latFrac = (latRad(ne.lat()) - latRad(sw.lat())) / Math.PI / 2;
          var lngDiff = ne.lng() - sw.lng();
          if (lngDiff < 0) lngDiff += 360;
          var lngFrac = lngDiff / 360;
          if (latFrac <= 0 || lngFrac <= 0) return gmap.getZoom();
          var latZoom = Math.log(h / 256 / latFrac) / Math.LN2;
          var lngZoom = Math.log(w / 256 / lngFrac) / Math.LN2;
          return Math.max(0, Math.floor(Math.min(latZoom, lngZoom)));
        },

        addLayer: function (layer) {
          if (layer && layer.addTo) layer.addTo(this);
          else if (layer && layer.setMap) layer.setMap(gmap);
          if (this._layers.indexOf(layer) === -1) this._layers.push(layer);
          return this;
        },
        removeLayer: function (layer) {
          if (!layer) return this;
          if (layer.remove) layer.remove();
          else if (layer.setMap) layer.setMap(null);
          var i = this._layers.indexOf(layer);
          if (i >= 0) this._layers.splice(i, 1);
          return this;
        },
        hasLayer: function (layer) { return this._layers.indexOf(layer) >= 0; },

        // applyRadarModeSemantics() walks the layers to strip RainViewer tiles
        // when switching to historical mode, so this iterates the real list.
        eachLayer: function (fn) {
          this._layers.slice().forEach(function (l) { fn(l); });
          return this;
        },

        // Every InfoWindow opened through the shim registers here so one
        // closePopup() clears them all, as Leaflet map.closePopup() does.
        _registerPopup: function (info) {
          if (this._openInfoWindows.indexOf(info) === -1) this._openInfoWindows.push(info);
        },
        closePopup: function () {
          this._openInfoWindows.forEach(function (i) { try { i.close(); } catch (e) {} });
          this._analysisPopupOpen = false;
          return this;
        },

        // Leaflet needs invalidateSize() after a container resize (tab switch).
        // Google relayouts on its 'resize' event; re-centring afterwards keeps
        // the view anchored exactly as the Leaflet call sites intend.
        invalidateSize: function () {
          var c = gmap.getCenter();
          google.maps.event.trigger(gmap, 'resize');
          if (c) gmap.setCenter(c);
          return this;
        },

        remove: function () { return this; }
      };

      // Restore the HUD Google just cleared, stacked above the map surface.
      // pointer-events are left to each element's own classes so the map stays
      // draggable underneath the transparent HUD wrapper regions.
      hud.forEach(function (node) {
        if (!node.style.zIndex) node.style.zIndex = '500';
        el.appendChild(node);
      });

      // Clicking the base map closes open popups, matching Leaflet.
      gmap.addListener('click', function () { lMap.closePopup(); });
      return lMap;
    },

    // ── Tile layer ───────────────────────────────────────────────────────────
    // Previously a no-op, which silently dropped both the Esri base tiles and
    // the RainViewer radar mosaic. Implemented as a real Google ImageMapType
    // overlay so the radar product actually renders.
    tileLayer: function (urlTemplate, options) {
      options = options || {};
      var layer = {
        _url: urlTemplate,
        _gmap: null,
        _imageType: null,
        _opacity: options.opacity != null ? options.opacity : 1.0,
        // OVERZOOM (CHANGE 4).
        //
        // RainViewer publishes radar only to maxNativeZoom (7). This used to
        // `return null` above that zoom, so the radar VANISHED the moment the
        // user zoomed past 7 -- the reported "radar disappears when zooming in".
        //
        // Leaflet's behaviour, reproduced here, is to keep displaying the
        // deepest available tile, magnified. Google's ImageMapType cannot
        // upscale, so tiles are built as DOM nodes: beyond maxNativeZoom we
        // fetch the ANCESTOR tile at maxNativeZoom and position/scale it with
        // CSS so the correct quarter of it fills the requested tile. The
        // imagery stays geographically aligned at every zoom; it simply gets
        // blockier, which is honest -- no radar detail is invented.
        _build: function () {
          var self = this;
          var maxNative = options.maxNativeZoom;

          return {
            tileSize: new google.maps.Size(256, 256),
            maxZoom: options.maxZoom || 18,
            name: options.attribution || '',
            getTile: function (coord, zoom, ownerDocument) {
              var div = ownerDocument.createElement('div');
              div.style.width = '256px';
              div.style.height = '256px';
              div.style.overflow = 'hidden';
              div.style.position = 'relative';
              if (coord == null) return div;

              var n = 1 << zoom;
              if (coord.y < 0 || coord.y >= n) return div;
              var x = ((coord.x % n) + n) % n;
              var y = coord.y;
              var z = zoom;
              var scale = 1, offX = 0, offY = 0;

              if (maxNative != null && zoom > maxNative) {
                // Walk up to the deepest published ancestor tile.
                var dz = zoom - maxNative;
                var factor = 1 << dz;             // tiles per ancestor per axis
                var ax = Math.floor(x / factor);
                var ay = Math.floor(y / factor);
                scale = factor;                    // magnification
                // Which sub-tile of the ancestor this request represents.
                offX = (x - ax * factor) * 256;
                offY = (y - ay * factor) * 256;
                x = ax; y = ay; z = maxNative;
              }

              var url = urlTemplate
                .replace('{z}', z)
                .replace('{x}', x)
                .replace('{y}', y)
                .replace('{s}', 'a');

              var img = ownerDocument.createElement('img');
              img.src = url;
              img.style.position = 'absolute';
              img.style.width = (256 * scale) + 'px';
              img.style.height = (256 * scale) + 'px';
              img.style.left = (-offX) + 'px';
              img.style.top = (-offY) + 'px';
              img.style.imageRendering = 'auto';
              img.style.pointerEvents = 'none';
              // A missing frame must not paint a broken-image glyph over the map.
              img.onerror = function () { img.style.display = 'none'; };
              div.appendChild(img);

              div.style.opacity = String(self._opacity);
              return div;
            },
            releaseTile: function (tile) {
              if (!tile) return;
              var imgs = tile.getElementsByTagName('img');
              for (var i = 0; i < imgs.length; i++) {
                var img = imgs[i];
                // Do NOT wipe src while the image is still downloading.
                //
                // Google calls releaseTile() for tiles it is recycling, and it
                // does so even for tiles whose <img> has not finished loading.
                // Assigning src = '' to an in-flight <img> CANCELS the request,
                // which surfaced as exactly 9 net::ERR_ABORTED entries in the
                // browser console on every visit to the Radar view (proven:
                // 9 aborts == 9 src-wipes, and the map immediately re-requested
                // the same 9 URLs and got HTTP 200). Nothing was visually
                // broken, but a real request was being started and killed, and
                // genuine radar failures were indistinguishable from this noise.
                //
                // `complete` is true once the fetch has finished (successfully
                // or not), so releasing only completed images frees the decoded
                // bitmap without aborting anything. An incomplete image is left
                // alone; it is detached from the DOM with its parent tile and
                // is garbage-collected normally.
                if (img.complete) {
                  img.removeAttribute('src');
                }
              }
            }
          };
        },
        addTo: function (lMap) {
          var gm = (lMap && lMap._gmap) ? lMap._gmap : lMap;
          this._gmap = gm;
          this._imageType = this._build();
          gm.overlayMapTypes.push(this._imageType);
          if (lMap && lMap._layers && lMap._layers.indexOf(this) === -1) {
            lMap._layers.push(this);
          }
          return this;
        },
        setOpacity: function (o) {
          this._opacity = o;
          // The overzoom-capable map type is a plain object with getTile(), not
          // a google.maps.ImageMapType, so it has no setOpacity(). Opacity is
          // applied per tile from _opacity; already-rendered tiles are updated
          // in place so the change is immediate rather than waiting for a pan.
          if (this._imageType && typeof this._imageType.setOpacity === 'function') {
            this._imageType.setOpacity(o);
          } else if (this._gmap && this._gmap.getDiv()) {
            var host = this._gmap.getDiv();
            var tiles = host.querySelectorAll('div[style*="256px"]');
            for (var i = 0; i < tiles.length; i++) {
              if (tiles[i].getElementsByTagName('img').length) {
                tiles[i].style.opacity = String(o);
              }
            }
          }
          return this;
        },
        setMap: function (m) { if (m === null) this.remove(); return this; },
        remove: function () {
          if (this._gmap && this._imageType) {
            var arr = this._gmap.overlayMapTypes;
            for (var i = 0; i < arr.getLength(); i++) {
              if (arr.getAt(i) === this._imageType) { arr.removeAt(i); break; }
            }
          }
          this._imageType = null;
          return this;
        },
        // Google has no per-overlay z-order control for map types.
        bringToFront: function () { return this; },
        bringToBack: function () { return this; }
      };
      return layer;
    },

    // ── Image overlay (StormSense risk / observation / historical surfaces) ──
    imageOverlay: function (url, bounds, options) {
      options = options || {};
      var gb = toGBounds(bounds);
      var rect = {
        south: gb.getSouthWest().lat(), west: gb.getSouthWest().lng(),
        north: gb.getNorthEast().lat(), east: gb.getNorthEast().lng()
      };
      var opacity = options.opacity != null ? options.opacity : 1.0;
      var overlay = new google.maps.GroundOverlay(url, rect, {
        opacity: opacity,
        clickable: !!options.interactive
      });

      return {
        _bounds: gb,
        addTo: function (lMap) {
          overlay.setMap(lMap._gmap);
          if (lMap._layers && lMap._layers.indexOf(this) === -1) lMap._layers.push(this);
          return this;
        },
        setMap: function (m) { overlay.setMap(m); return this; },
        remove: function () { overlay.setMap(null); return this; },
        getBounds: function () { return wrapBounds(gb); },
        // GroundOverlay has no setUrl, so swap in a new one over the same rect
        // and re-attach it to whatever map the old one was on.
        setUrl: function (newUrl) {
          var m = overlay.getMap();
          overlay.setMap(null);
          overlay = new google.maps.GroundOverlay(newUrl, rect, {
            opacity: opacity, clickable: !!options.interactive
          });
          if (m) overlay.setMap(m);
          return this;
        },
        setOpacity: function (o) { opacity = o; overlay.setOpacity(o); return this; },
        // Google draws GroundOverlays in a fixed pane; no z-order control.
        bringToFront: function () { return this; },
        bringToBack: function () { return this; }
      };
    },

    // ── GeoJSON (state boundary, districts, inspection grid) ─────────────────
    // Uses an independent google.maps.Data instance per layer. The previous
    // version called map.data.setStyle(), so the LAST geoJSON layer added
    // restyled every earlier one, and onEachFeature was ignored entirely --
    // district tooltips and grid-cell popups never bound at all.
    geoJSON: function (data, options) {
      options = options || {};
      var dataLayer = new google.maps.Data();
      var features = data ? dataLayer.addGeoJson(data) : [];
      var overrides = {};   // feature id -> style applied by setStyle() on hover
      var lMapRef = null;
      var infoWindow = null;   // click popups (analysis)
      var tipWindow = null;    // hover labels (district names) -- kept separate
      var bounds = new google.maps.LatLngBounds();

      features.forEach(function (f) {
        var geom = f.getGeometry();
        if (geom) geom.forEachLatLng(function (ll) { bounds.extend(ll); });
      });

      function propsOf(feature) {
        var props = {};
        feature.forEachProperty(function (v, k) {
          if (k.indexOf('__') !== 0) props[k] = v;
        });
        return props;
      }

      function baseStyleFor(feature) {
        var s = options.style;
        if (typeof s === 'function') {
          s = s({properties: propsOf(feature), type: 'Feature'});
        }
        var g = toGStyle(s);
        if (options.interactive === false) g.clickable = false;
        return g;
      }

      dataLayer.setStyle(function (feature) {
        var id = feature.getId();
        var base = baseStyleFor(feature);
        if (id != null && overrides[id]) {
          Object.keys(overrides[id]).forEach(function (k) { base[k] = overrides[id][k]; });
        }
        return base;
      });

      // Leaflet per-feature layer handle, backed by the real Data feature so
      // bindTooltip/bindPopup/setStyle/on all act on the drawn geometry.
      function featureHandle(feature) {
        return {
          _feature: feature,
          feature: {properties: propsOf(feature), type: 'Feature'},
          bindPopup: function (html) { feature.setProperty('__popup', html); return this; },
          bindTooltip: function (html) { feature.setProperty('__tooltip', html); return this; },
          setStyle: function (st) {
            var id = feature.getId();
            var g = toGStyle(st);
            if (id != null) overrides[id] = g;
            dataLayer.overrideStyle(feature, g);
            return this;
          },
          on: function (ev, fn) {
            var self = this;
            if (['mouseover', 'mouseout', 'click'].indexOf(ev) === -1) return this;
            dataLayer.addListener(ev, function (e) {
              if (e.feature === feature) fn.call(self, e);
            });
            return this;
          }
        };
      }

      if (typeof options.onEachFeature === 'function') {
        features.forEach(function (f) {
          options.onEachFeature({properties: propsOf(f), type: 'Feature'}, featureHandle(f));
        });
      }

      // Popups and hover tooltips get SEPARATE InfoWindows.
      //
      // They previously shared one: moving the cursor across a district then
      // overwrote the risk-analysis popup with the district's name tooltip, so
      // a click that had correctly fetched and rendered point forecast values
      // ended up displaying just "Birbhum". A hover label must never be able to
      // destroy the analysis the user deliberately clicked for.
      dataLayer.addListener('click', function (e) {
        var html = e.feature.getProperty('__popup');
        if (!html || !lMapRef) return;
        if (tipWindow) tipWindow.close();
        if (!infoWindow) infoWindow = new google.maps.InfoWindow();
        infoWindow.setContent(html);
        infoWindow.setPosition(e.latLng);
        infoWindow.open(lMapRef._gmap);
        lMapRef._registerPopup(infoWindow);
      });
      dataLayer.addListener('mousemove', function (e) {
        var tip = e.feature.getProperty('__tooltip');
        if (!tip || !lMapRef) return;
        // Never raise a hover label over an open analysis popup.
        if (lMapRef._analysisPopupOpen) return;
        if (!tipWindow) {
          tipWindow = new google.maps.InfoWindow({disableAutoPan: true});
        }
        tipWindow.setContent(tip);
        tipWindow.setPosition(e.latLng);
        tipWindow.open(lMapRef._gmap);
      });
      dataLayer.addListener('mouseout', function () {
        if (tipWindow) tipWindow.close();
      });

      return {
        _dataLayer: dataLayer,
        addTo: function (lMap) {
          lMapRef = lMap;
          dataLayer.setMap(lMap._gmap);
          if (lMap._layers && lMap._layers.indexOf(this) === -1) lMap._layers.push(this);
          return this;
        },
        setMap: function (m) { dataLayer.setMap(m); return this; },
        remove: function () { dataLayer.setMap(null); return this; },
        // initMap()/updateMapDomain() frame the view on the real state outline.
        getBounds: function () { return wrapBounds(bounds); },
        eachLayer: function (fn) { features.forEach(function (f) { fn(featureHandle(f)); }); },
        setStyle: function (st) { dataLayer.setStyle(toGStyle(st)); return this; },
        bringToFront: function () { return this; },
        bringToBack: function () { return this; }
      };
    },

    // ── Marker ───────────────────────────────────────────────────────────────
    marker: function (latlng, options) {
      options = options || {};
      var opts = {position: toLatLng(latlng)};
      var icon = options.icon;

      if (icon && icon.html) {
        // L.divIcon carries arbitrary HTML. Google Markers take an image, so
        // the HTML is wrapped in an SVG foreignObject data URI, preserving the
        // caption/beacon markup the app supplies instead of discarding it.
        var w = (icon.iconSize && icon.iconSize[0]) || 40;
        var h = (icon.iconSize && icon.iconSize[1]) || 40;
        var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h + '">'
                + '<foreignObject width="100%" height="100%">'
                + '<div xmlns="http://www.w3.org/1999/xhtml">' + icon.html + '</div>'
                + '</foreignObject></svg>';
        opts.icon = {
          url: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg),
          scaledSize: new google.maps.Size(w, h),
          anchor: new google.maps.Point(w / 2, h / 2)
        };
      }
      if (options.zIndexOffset != null) opts.zIndex = options.zIndexOffset;

      var marker = new google.maps.Marker(opts);
      var lMapRef = null;

      marker.addTo = function (lMap) {
        lMapRef = lMap;
        this.setMap(lMap._gmap);
        if (lMap._layers && lMap._layers.indexOf(this) === -1) lMap._layers.push(this);
        return this;
      };
      marker.bindPopup = function (html) {
        var info = new google.maps.InfoWindow({content: html});
        this._info = info;
        marker.addListener('click', function () {
          info.open(marker.getMap(), marker);
          if (lMapRef) lMapRef._registerPopup(info);
        });
        return this;
      };
      marker.openPopup = function () {
        if (this._info && this.getMap()) {
          this._info.open(this.getMap(), this);
          if (lMapRef) lMapRef._registerPopup(this._info);
        }
        return this;
      };
      marker.closePopup = function () { if (this._info) this._info.close(); return this; };
      marker.setLatLng = function (ll) { this.setPosition(toLatLng(ll)); return this; };
      marker.getLatLng = function () { return this.getPosition(); };
      marker.remove = function () { this.setMap(null); return this; };
      return marker;
    },

    // ── Circle marker (observation stations) ─────────────────────────────────
    // Leaflet circleMarker radius is in SCREEN PIXELS. Modelling it as a
    // metre-based google.maps.Circle made station dots resize with zoom and, at
    // radius*1000, drew them kilometres wide. A scaled symbol keeps the pixel
    // semantics the call site was written for.
    circleMarker: function (latlng, options) {
      options = options || {};
      var marker = new google.maps.Marker({
        position: toLatLng(latlng),
        icon: {
          path: google.maps.SymbolPath.CIRCLE,
          scale: options.radius || 8,
          fillColor: options.fillColor || options.color || '#38bdf8',
          fillOpacity: options.fillOpacity != null ? options.fillOpacity : 0.9,
          strokeColor: options.color || '#ffffff',
          strokeWeight: options.weight != null ? options.weight : 1,
          strokeOpacity: options.opacity != null ? options.opacity : 1
        }
      });
      var lMapRef = null;
      marker.addTo = function (lMap) {
        lMapRef = lMap;
        this.setMap(lMap._gmap);
        if (lMap._layers && lMap._layers.indexOf(this) === -1) lMap._layers.push(this);
        return this;
      };
      marker.bindPopup = function (html) {
        var info = new google.maps.InfoWindow({content: html});
        this._info = info;
        marker.addListener('click', function () {
          info.open(marker.getMap(), marker);
          if (lMapRef) lMapRef._registerPopup(info);
        });
        return this;
      };
      marker.openPopup = function () {
        if (this._info && this.getMap()) this._info.open(this.getMap(), this);
        return this;
      };
      marker.closePopup = function () { if (this._info) this._info.close(); return this; };
      marker.setStyle = function (st) {
        var ic = this.getIcon() || {};
        if (st.fillColor) ic.fillColor = st.fillColor;
        if (st.color) ic.strokeColor = st.color;
        if (st.weight != null) ic.strokeWeight = st.weight;
        if (st.fillOpacity != null) ic.fillOpacity = st.fillOpacity;
        this.setIcon(ic);
        return this;
      };
      marker.remove = function () { this.setMap(null); return this; };
      return marker;
    },

    // ── Standalone popup ─────────────────────────────────────────────────────
    popup: function (options) {
      var info = new google.maps.InfoWindow(options || {});
      return {
        _info: info,
        setLatLng: function (ll) { info.setPosition(toLatLng(ll)); return this; },
        setContent: function (c) { info.setContent(c); return this; },
        openOn: function (lMap) {
          info.open(lMap._gmap);
          if (lMap._registerPopup) lMap._registerPopup(info);
          // Marks an ANALYSIS popup as owning the map's popup surface, so a
          // district hover label cannot replace it (see the Data-layer
          // mousemove handler). Cleared when the popup closes.
          lMap._analysisPopupOpen = true;
          google.maps.event.addListenerOnce(info, 'closeclick', function () {
            lMap._analysisPopupOpen = false;
          });
          return this;
        },
        close: function () { info.close(); return this; },
        setMap: function (m) { if (m === null) info.close(); return this; }
      };
    },

    latLngBounds: function (a, b) {
      // Accepts (corner, corner) or a single [[s,w],[n,e]] array.
      if (b === undefined) return wrapBounds(toGBounds(a));
      return wrapBounds(toGBounds([a, b]));
    },
    latLng: function (lat, lng) { return {lat: lat, lng: lng}; },

    divIcon: function (options) { return options || {}; },
    icon: function (options) { return options || {}; },

    layerGroup: function () {
      var layers = [];
      var lMapRef = null;
      return {
        _layers: layers,
        addLayer: function (layer) {
          layers.push(layer);
          if (lMapRef && layer.addTo) layer.addTo(lMapRef);
          return this;
        },
        clearLayers: function () {
          layers.forEach(function (l) { if (l.setMap) l.setMap(null); });
          layers.length = 0;
          return this;
        },
        eachLayer: function (fn) { layers.slice().forEach(fn); return this; },
        addTo: function (lMap) {
          lMapRef = lMap;
          layers.forEach(function (l) { if (l.addTo) l.addTo(lMap); });
          if (lMap._layers && lMap._layers.indexOf(this) === -1) lMap._layers.push(this);
          return this;
        },
        setMap: function (m) {
          layers.forEach(function (l) { if (l.setMap) l.setMap(m); });
          return this;
        },
        remove: function () { return this.clearLayers(); }
      };
    }
  };
})();
// ──────────────────────────────────────────────────────────────────────────
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
  // The dashboard opens on NOW, not on a forecast horizon: the first thing a
  // viewer should see is the current risk state, with +2/+4/+6h as deliberate
  // follow-ups. Must be the STRING "now" (not 0) -- every horizon-aware branch
  // in this file tests `currentLeadHours === "now"` to decide whether to paint
  // observations or a forecast, and apiLeadHours() maps "now" -> lead 0 for the
  // API. Using 0 here would request the right data but take the forecast
  // branches, labelling live observations as a forecast.
  window.currentLeadHours = "now";
  // Benchmark lead is independent: the WRF/Nowcast comparison has no NOW column
  // (the model's heads start at +2h), so it stays on its own default.
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
  //
  // There is exactly ONE refresh timer and ONE countdown ticker for the whole
  // dashboard. `nextRefreshAt` is the single source of truth for both: the
  // countdown is derived from it rather than decremented independently, so the
  // number on screen can never drift away from when the refresh actually fires.
  var LIVE_REFRESH_MS = 300000;
  var nextRefreshAt = Date.now() + LIVE_REFRESH_MS;
  var liveRefreshTimer = null;
  var countdownTimer = null;

  /**
   * The lead hour to send to the API.
   *
   * `window.currentLeadHours` is the string 'now' while the NOW horizon is
   * selected, because NOW is an OBSERVATION view rather than a forecast lead.
   * The backend only accepts numeric leads, so passing 'now' through produced
   * HTTP 422s and silently broke the background refresh while NOW was selected.
   * NOW still needs the forecast products underneath (cards, districts,
   * bulletins), so it resolves to the shortest real horizon.
   */
  function apiLeadHours() {
    var lead = window.currentLeadHours;
    if (lead === "now" || lead === "NOW" || lead == null) return 0;
    var n = Number(lead);
    return (n === 0 || n === 2 || n === 3 || n === 4 || n === 5 || n === 6) ? n : 0;
  }

  function formatCountdown(totalSeconds) {
    if (totalSeconds < 0) totalSeconds = 0;
    var m = Math.floor(totalSeconds / 60);
    var s = totalSeconds % 60;
    return m > 0 ? m + "m " + (s < 10 ? "0" : "") + s + "s" : s + "s";
  }

  // Restarts the countdown window. Called after every completed refresh
  // (automatic or manual) so a manual refresh genuinely resets the cadence.
  function resetRefreshCountdown() {
    nextRefreshAt = Date.now() + LIVE_REFRESH_MS;
    renderCountdown();
  }

  function renderCountdown() {
    var el = document.getElementById("live-refresh-countdown");
    if (!el) return;
    if (window.stormSenseMode !== "live") {
      el.textContent = "paused (historical)";
      return;
    }
    var remaining = Math.max(0, Math.round((nextRefreshAt - Date.now()) / 1000));
    el.textContent = formatCountdown(remaining);
  }

  // The monitored region is the whole state; the active high-risk area is
  // computed from the same forecast the map and cards show. Nothing here is
  // hardcoded to a district -- contradictory region labels were a real defect.
  //
  // NOTE: this must NOT write the region name into #profile-district. That
  // element is the CURRENT LOCATION heading and is owned by applyLiveLocation();
  // overwriting it here made the panel say "West Bengal" while simultaneously
  // reporting that the current location was unavailable.
  /**
   * Keep the hazard cards' scope line truthful for the SELECTED AREA.
   *
   * The three cards were hardcoded to "... Hazard · West Bengal peak". Once an
   * area is selected the backend aggregates over that district only, so the
   * label would otherwise claim a state-wide peak while showing one district's
   * value. Text only -- no metric, value, unit or status indicator changes.
   */
  function applyAreaScopeLabels() {
    var d = window.activeDistrict();
    var scope = (d && d !== "West Bengal") ? d : "West Bengal peak";
    [["kpi-ts-scope", "Atmospheric Hazard"],
     ["kpi-rain-scope", "Precipitation Hazard"],
     ["kpi-ff-scope", "Hydrological Hazard"]].forEach(function (pair) {
      var el = document.getElementById(pair[0]);
      if (el) el.textContent = pair[1] + " · " + scope;
    });
    var areaEl = document.getElementById("selected-area-name");
    if (areaEl) areaEl.textContent = (d && d !== "West Bengal") ? d : "Whole State";
  }
  window.applyAreaScopeLabels = applyAreaScopeLabels;

  function applyRegionState(summary) {
    applyAreaScopeLabels();
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
    district: "West Bengal",
    state: "West Bengal",
    country: "India",
    shortLabel: "West Bengal",
    lat: 22.724,
    lon: 88.479,
    subdivisions: ["Barasat Sadar", "Barrackpore", "Bangaon", "Basirhat", "Bidhannagar"],
    defaultSector: "Barasat Sadar"
  };

  // -----------------------------------------------------------------------------
  // Dynamic Time Formatters
  // -----------------------------------------------------------------------------
  function parseUtcIso(isoStr) {
    if (!isoStr) return new Date("2024-05-26T12:00:00Z");
    var d = new Date(isoStr);
    return isNaN(d.getTime()) ? new Date("2024-05-26T12:00:00Z") : d;
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

  /**
   * Renders the reference instant and its exact forecast valid time.
   *
   * `issueTimeStr` is THE authoritative reference instant supplied by the
   * backend (exact wall-clock in live mode, never floored). The valid time is
   * that instant plus exactly leadHours -- computed from the same reference the
   * API used, so the browser and the API can never disagree.
   *
   * `analysisTimeStr`, when present, is the real GFS analysis behind the input
   * state. It is a DIFFERENT instant and is displayed separately; it must never
   * be presented as the issue or valid time.
   */
  function updateDynamicTimes(issueTimeStr, leadHours, analysisTimeStr) {
    // NOW IS LEAD ZERO. At t=0 the valid time IS the analysis/issue time --
    // there is no forecast offset. This previously coerced 'now' to 2 (and the
    // caller could pass a stale numeric lead), so selecting NOW displayed a
    // forecast valid time hours ahead of the state actually on screen: the
    // historical case study showed "Valid 18:00 UTC" over its 12:00 analysis.
    var isNow = (window.currentLeadHours === "now" || leadHours === "now" || leadHours === 0);
    leadHours = isNow ? 0 : (Number(leadHours) || 2);
    var issueDate = parseUtcIso(issueTimeStr);
    var validDate = new Date(issueDate.getTime() + leadHours * 3600 * 1000);

    var issueFormatted = formatUtcDateTime(issueDate);
    var validFormatted = formatUtcDateTime(validDate);
    var leadFormatted = isNow
      ? "Analysis time (no forecast offset)"
      : "+" + leadHours + "h (" + (leadHours * 60) + "m Lead)";

    document.querySelectorAll("#bind-issue-time, .bind-issue-time").forEach(function (el) { el.textContent = issueFormatted; });
    document.querySelectorAll("#bind-forecast-lead, .bind-forecast-lead").forEach(function (el) { el.textContent = leadFormatted; });
    document.querySelectorAll("#bind-valid-time, .bind-valid-time").forEach(function (el) { el.textContent = validFormatted; });
    var fvValid = document.getElementById("fv-valid-time");
    var fvGfs = document.getElementById("fv-gfs-time");
    var fvGfsWrap = document.getElementById("fv-gfs-run-container");
    if (fvValid) fvValid.textContent = validFormatted;

    // PROMINENT TEMPORAL CONTRACT.
    // Both halves derive from the SAME authoritative reference instant the API
    // supplied: the valid time is reference + leadHours, and the offset is
    // stated explicitly so "+4h" can never be read next to a +2h timestamp.
    // IST is primary because the monitored region is West Bengal.
    setText("fv-horizon-title", "+" + leadHours + "H FORECAST");
    setText("fv-valid-ist", formatIstStamp(validDate));
    setText("fv-offset-from-now", "+" + leadHours + ":00 FROM NOW");

    // Show the ANALYSIS time, not the issue time. Showing the issue time here
    // previously implied GFS had published a cycle at the current instant.
    if (fvGfs) {
      if (analysisTimeStr) {
        fvGfs.textContent = formatUtcDateTime(parseUtcIso(analysisTimeStr));
        if (fvGfsWrap) fvGfsWrap.style.display = "";
      } else if (fvGfsWrap) {
        fvGfsWrap.style.display = "none";
      }
    }

    var deskLead = document.getElementById("desk-lead-indicator");
    // In historical mode this strip sits under the OBSERVED t=0 analysis field,
    // which has no lead time; "+2h Calibrated Horizon" there would label an
    // observation as a forecast.
    if (deskLead) {
      deskLead.textContent = (window.stormSenseMode === "historical")
        ? "Analysis t=0 · observed"
        : "+" + leadHours + "h Calibrated Horizon";
    }

    var deskCellsLead = document.getElementById("desk-cells-lead-tag");
    if (deskCellsLead) deskCellsLead.textContent = "+" + leadHours + "h Horizon";

    // Do NOT stamp a "+Nh" tag over the map legend while NOW is the selected
    // horizon. This function is called on every live refresh with a coerced
    // numeric lead, so it used to silently relabel the NOW map as "+2h" -- the
    // map correctly showed no forecast field while the badge claimed a forecast.
    var legendLead = document.getElementById("legend-lead-tag");
    if (legendLead && window.currentLeadHours !== "now") {
      legendLead.textContent = "+" + leadHours + "h";
    }
  }

  /**
   * Format an instant as a prominent IST stamp: "12 SEP · 00:09 IST".
   *
   * Uses UTC+5:30 arithmetic on the absolute epoch value, so date, month and
   * year rollovers fall out correctly and the result never depends on the
   * viewer's own timezone.
   */
  function formatIstStamp(date) {
    var ist = new Date(date.getTime() + (5.5 * 3600000));
    var months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
    return String(ist.getUTCDate()).padStart(2, "0") + " "
      + months[ist.getUTCMonth()] + " · "
      + String(ist.getUTCHours()).padStart(2, "0") + ":"
      + String(ist.getUTCMinutes()).padStart(2, "0") + " IST";
  }

  function formatIstClock(date) {
    var utc = date.getTime() + (date.getTimezoneOffset() * 60000);
    var ist = new Date(utc + (5.5 * 3600000));
    return String(ist.getHours()).padStart(2, "0") + ":" +
           String(ist.getMinutes()).padStart(2, "0") + " IST";
  }

  // -----------------------------------------------------------------------------
  // Real-Time IST Clock (Every 1 Second)
  // -----------------------------------------------------------------------------
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

  // -----------------------------------------------------------------------------
  // Live Surface Observation & Data Health Ingest (Every 60 Seconds)
  // -----------------------------------------------------------------------------
  window.fetchLiveSurfaceData = function (isManual) {
    return fetch(API_BASE + "/api/live/surface" + areaCoordQS())
      .then(parseJson)
      .then(function (data) {
        if (!data) return;
        window.StormSenseLiveSurface = data;

        // OBSERVED TIMESTAMP MUST NOT GO BACKWARDS (CHANGE 5).
        //
        // The upstream station API can return slightly different observation
        // timestamps between calls (it serves whichever scan its edge has).
        // Measured: a refresh replaced an 06:30:05 observation with 06:25:26,
        // so "OBSERVED: 2m ago" jumped BACKWARDS to "7m ago" -- the display
        // aged instead of refreshing.
        //
        // The newest observation actually seen wins. This never invents a
        // timestamp: it only refuses to adopt an older one than is already
        // displayed, so the header stays monotonic and consistent with the
        // clock. A genuinely newer scan is adopted immediately.
        var incoming = data.observed_at_utc ? Date.parse(data.observed_at_utc) : NaN;
        var held = window.currentObsTime ? Date.parse(window.currentObsTime) : NaN;
        if (!isNaN(incoming) && (isNaN(held) || incoming >= held)) {
          window.currentObsTime = data.observed_at_utc;
        } else if (isNaN(held)) {
          window.currentObsTime = data.observed_at_utc;
        }
        // Re-stamp the header immediately rather than waiting for the next
        // 1s clock tick, so a manual Refresh shows its effect at once.
        if (typeof updateIstClock === "function") updateIstClock();

        var obs = data.observations || {};

        if (window.stormSenseMode === "live") {
          paintDashboardLive(obs, data);
        }

        // NOTE: the countdown is owned solely by resetRefreshCountdown(), which
        // the refresh cycle calls on completion. Resetting it here as well used
        // to make the displayed countdown disagree with the actual timer.
        if (isManual) {
          window.showToast("Live observation refreshed", "check_circle");
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

    var lead = apiLeadHours();
    var modeQS = "&mode=live";

    return Promise.all([
      window.fetchLiveSurfaceData(false).catch(function() { return null; }),
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + modeQS + districtQS()).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/districts?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live") + "&lead=" + apiLeadHours()).then(parseJson).catch(function () { return null; }),
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
          if (summary.issue_time) updateDynamicTimes(summary.issue_time, lead, summary.analysis_time_iso);
        }
        applyRegionState(summary);
        applyFreshnessState(summary);

        if (districts && districts.length) {
          window.StormSenseDistrictsData = districts;
          renderDistricts(adaptDistricts(districts, lead, summary && summary.issue_time));
          // Bulletins come from the SAME district aggregation the map uses.
          window.renderBulletinsFromDistricts(districts, summary);
        } else {
          window.renderBulletinsFromDistricts([], summary);
        }

        if (cells) {
          window.StormSenseHighRiskCells = cells;
          renderHighRiskCells(cells);
          if (window.stormSenseMap) renderHotspotBeacons(window.stormSenseMap, cells);
        }

        // Repaint the surface for the CURRENT horizon; never re-fit the map, so
        // the operator's pan/zoom survives the refresh.
        // Pass the RAW horizon (which may be 'now'), not apiLeadHours(): the
        // latter coerces 'now' to 2, which would let the auto-refresh quietly
        // paint the +2h forecast field while NOW is selected.
        if (window.stormSenseMap) {
          renderContinuousRiskSurface(window.stormSenseMap, window.currentLeadHours);
        }

        setText("last-updated-time", formatIstClock(new Date()));
        // A completed refresh cycle restarts the countdown window, so the
        // displayed countdown always reflects time-until-next-refresh.
        resetRefreshCountdown();
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

  // -----------------------------------------------------------------------------
  // Operational Mode Controller (LIVE vs HISTORICAL)
  // -----------------------------------------------------------------------------
  window.switchMode = function (targetMode) {
    if (targetMode !== "live" && targetMode !== "historical") targetMode = "live";
    window.stormSenseMode = targetMode;

    if (window.history && window.history.pushState) {
      var newUrl = targetMode === "historical" ? "?mode=historical" : window.location.pathname;
      window.history.pushState({ mode: targetMode }, "", newUrl);
    }

    updateModeUI(targetMode);
    if (targetMode === "historical") {
        // NOW is now fully supported in historical mode: it renders the event's
        // t=0 ANALYSIS field (/api/historical/analysis-surface). The horizon is
        // therefore preserved across the mode switch instead of being forced
        // to +2h. Only a genuinely unset horizon falls back.
        // Entering the case study starts at its ANALYSIS state (T0), which is
        // the event's own "now". Inheriting the previously selected live
        // horizon left the card showing a stale "+6h" with live timestamps.
        var lead = (window.currentLeadHours == null) ? "now" : window.currentLeadHours;
        window.currentLeadHours = lead;
        loadDashboardData().then(function (data) {
            // Paint the HISTORICAL data that was just fetched. This previously
            // called refreshLiveDashboard(), which repaints from the LIVE path
            // -- so the historical t=0 values fetched here were never rendered
            // and the conditions panel stayed on em dashes.
            if (data) paintDashboard(data);
            updateMapMode("historical");
            // Repaint the conditions strip from the EVENT's t=0 reanalysis.
            // Nothing else owns this strip in historical mode, so without it
            // the live values from before the switch stayed on screen.
            if (window.StormSenseNowcastData) {
              renderNowcastHistorical(window.StormSenseNowcastData);
            }
            // Re-bind the Current Location panel to the CASE STUDY.
            // /api/nowcast/point accepts ?mode=historical and returns Remal's
            // values AT THE USER'S COORDINATES (Kolkata: 50.5% / 9.9 mm /
            // 23.3%), which is a different and more meaningful number than the
            // domain-wide peak (100% / 47.3 mm) the meters showed before.
            // Without this the location panel simply never followed the user
            // into historical mode.
            refreshLocationPanelForMode();
        });
    } else {
        window.fetchLiveSurfaceData(false);
        window.refreshLiveDashboard();
        updateMapMode("live");
        // Re-bind the Current Location panel to LIVE.
        //
        // The three risk meters and the horizon label are owned by
        // applyLiveLocation() -> applyPointRiskMeters(), which nothing called
        // on a mode switch. Leaving them alone meant Cyclone Remal's values
        // (100% / 47.3 mm / 100%) stayed on screen under a LIVE badge after
        // switching back -- historical data presented as current risk.
        refreshLocationPanelForMode();
    }
  };

  /**
   * Re-bind the Current Location panel (district, coordinates, the three risk
   * meters and the horizon label) to whichever mode is now active.
   *
   * Both modes are supported: /api/nowcast/point takes ?mode=, so historical
   * returns the case study's risk AT THE USER'S COORDINATES rather than the
   * domain-wide peak. Nothing here computes values in the frontend -- it only
   * re-queries with the correct mode so the panel stops describing the mode the
   * user just left.
   *
   * If geolocation has not resolved yet (or was denied) this acquires it once,
   * so entering historical mode directly still populates the panel.
   */
  function refreshLocationPanelForMode() {
    if (typeof window.applyLiveLocation !== "function") return;
    if (window.StormSenseUserLocation) {
      window.applyLiveLocation(window.StormSenseUserLocation);
      return;
    }
    if (typeof acquireLiveLocation === "function") {
      acquireLiveLocation().then(function (loc) {
        window.applyLiveLocation(loc);
      });
    } else {
      window.applyLiveLocation(null);
    }
  }

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
      if (modeBadgeLabel) modeBadgeLabel.textContent = "HISTORICAL CASE STUDY · CYCLONE REMAL";

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
        bannerDisclaimer.textContent = "StormSense ML Case Study · Cyclone Remal · analysis 26 May 2024 12:00 UTC · Grid: 0.25° (~28 km)";
      }

      if (deskModePill) {
        deskModePill.className = "px-2.5 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs font-bold uppercase tracking-wider font-mono flex items-center gap-1.5";
      }
      if (deskModeDot) deskModeDot.className = "size-2 rounded-full bg-cyan-400 animate-pulse";
      if (deskModeLabel) deskModeLabel.textContent = "HISTORICAL CASE STUDY · CYCLONE REMAL";
      if (deskSubtitle) {
        deskSubtitle.textContent = "Historical case study (0–6 h): Cyclone Remal, driven by the StormSense model from the 26 May 2024 12:00 UTC reanalysis state.";
      }

      // The single refresh control stays visible in historical mode; its
      // countdown reports that auto-refresh is paused (the case study is a
      // frozen state), and manual refresh simply re-pulls its products.
      if (refreshBox) refreshBox.style.display = "";
      renderCountdown();

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

      // The Current Location card describes OBSERVATIONS. In a frozen case
      // study there are no current observations -- the panel shows the event's
      // t=0 reanalysis state instead, and must say so. Leaving it reading
      // "Current Observations" with a live age ("just now") presented 2024 data
      // as today's weather.
      setText("location-obs-title", "Remal Historical Conditions · t=0");
      setText("location-obs-age", "analysis 26 May 2024 · 12:00 UTC");
      var obsDotH = document.getElementById("location-obs-dot");
      if (obsDotH) obsDotH.className = "size-1.5 rounded-full bg-cyan-400";
      var obsHeadH = document.getElementById("location-obs-heading");
      if (obsHeadH) obsHeadH.className = "text-[11px] font-bold font-mono text-cyan-300 uppercase tracking-wider flex items-center gap-1.5";
      // Per-field values and source labels are owned by
      // paintHistoricalConditions(), invoked from paintDashboard() once the
      // HISTORICAL summary has actually been fetched. Painting here would run
      // against the previous (live) summary, which carries no surface_obs_t0,
      // and would leave the panel reading "Unavailable" over real data.

      document.querySelectorAll(".kpi-mode-tag").forEach(function (el) {
        el.textContent = "CASE STUDY";
      });
      applyRadarModeSemantics();

      // Update map layers
      updateMapMode("historical");

      window.showToast("Active Mode: Historical Case Study (Cyclone Remal - Landfall Approach)", "history_edu");
    } else {
      // LIVE MODE
      if (pipeBadge) {
        pipeBadge.className = "flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px] font-mono font-bold";
      }
      if (pipeDot) pipeDot.className = "size-1.5 rounded-full bg-emerald-400 animate-pulse";
      // Describes the PIPELINE, not the selected horizon. It sits beside the
      // NOW/+2h/+4h/+6h controls, so "AI FORECAST ACTIVE" could be misread as a
      // claim that NOW is a forecast; name the pipeline explicitly instead.
      if (pipeText) pipeText.textContent = "LIVE PIPELINE ACTIVE";
      if (btn) {
        btn.className = "btn-mode-switch flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-bold transition-all shadow-md shadow-cyan-600/25 cursor-pointer border border-cyan-400/40";
      }
      if (btnLabel) btnLabel.textContent = "View Historical Case Study →";
      document.querySelectorAll(".kpi-mode-tag").forEach(function (el) {
        el.textContent = "LIVE";
      });
      applyRadarModeSemantics();
      // Restore the live observation labelling that historical mode retargets.
      setText("location-obs-title", "Current Observations");
      var obsDotL = document.getElementById("location-obs-dot");
      if (obsDotL) obsDotL.className = "size-1.5 rounded-full bg-emerald-400";
      var obsHeadL = document.getElementById("location-obs-heading");
      if (obsHeadL) obsHeadL.className = "text-[11px] font-bold font-mono text-emerald-300 uppercase tracking-wider flex items-center gap-1.5";
      ["bind-temp-source", "bind-rain-source", "bind-humidity-source", "bind-wind-source"].forEach(function (id) {
        setText(id, "Surface observation");
      });
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
        // Names the scope of each panel: the hazard cards are a state-wide
        // peak, the Current Location panel is the user's own coordinates.
        // Without this the peak district's number reads as a local value.
        deskSubtitle.textContent = "Hazard cards show the West Bengal-wide peak; the Current Location panel shows your coordinates. Updated every 5 minutes.";
      }

      if (refreshBox) refreshBox.style.display = "flex";
      resetRefreshCountdown();

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

  // -----------------------------------------------------------------------------
  // Data Loading & ViewModel Assembly
  // -----------------------------------------------------------------------------
  function parseJson(res) {
    if (!res || !res.ok) return null;
    return res.json().catch(function () { return null; });
  }

  function loadDashboardData() {
    var lead = apiLeadHours();
    var fetchCurrent = (window.stormSenseMode === "historical") 
        ? Promise.resolve(null) 
        : fetch(API_BASE + "/api/weather/current").then(parseJson).catch(function () { return null; });
    var fetchSurface = (window.stormSenseMode === "historical") 
        ? Promise.resolve(null) 
        : fetch(API_BASE + "/api/live/surface" + areaCoordQS()).then(parseJson).catch(function () { return null; });

    return Promise.all([
      fetchCurrent,
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + "&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/districts?lead=" + lead + "&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/risk-map?lead=" + lead + "&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live") + "&lead=" + apiLeadHours()).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/boundaries/west-bengal").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/boundaries/state").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/benchmark/models").then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8&mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; }),
      fetchSurface,
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
        // Bulletins must be populated on THIS path too (it is the one the
        // historical case study uses), from the same district aggregation.
        window.renderBulletinsFromDistricts(districts || [], summary);
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
      // The summary this viewmodel was built from. Painters that need fields
      // the viewmodel does not model (e.g. surface_obs_t0) read it from here
      // rather than from a global, which may not be assigned yet.
      rawSummary: summary,
      // NO FABRICATED FALLBACKS. These previously defaulted to invented values
      // (28.5 C / 32.0 / 80% / 16.0 km/h / "SE" / 0.0 mm) whenever the
      // observation fetch failed, so a dead API silently produced plausible
      // fake weather. Absent data stays null and renders as an em dash.
      // Rainfall in particular must never default to 0.0: absent data is not a
      // measurement of "no rain".
      currentConditions: {
        temperatureC: currentObs && currentObs.temperature != null ? currentObs.temperature : null,
        feelsLikeC: currentObs && currentObs.feels_like != null ? currentObs.feels_like : null,
        humidityPct: currentObs && currentObs.humidity != null ? currentObs.humidity : null,
        windKmh: currentObs && currentObs.wind_speed != null ? currentObs.wind_speed : null,
        windDirection: currentObs && currentObs.wind_direction != null
          ? getWindDirection(currentObs.wind_direction) : null,
        rainfallMmHr: currentObs
          ? (currentObs.rainfall_1h != null ? currentObs.rainfall_1h
             : (currentObs.rainfall_1h_mm != null ? currentObs.rainfall_1h_mm : null))
          : null
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
        deskLabel: "West Bengal desk",
        freshnessLabel: "LIVE INGEST · t=0"
      },
      disclaimer: "StormSense AI Forecast · AI-derived severe weather risk · Not an official government warning"
    };

    // Dynamic Time Derivation
    var issueTime = (summary && summary.issue_time) || "2024-05-26T12:00:00Z";
    var lead = (summary && summary.lead_hours) || window.currentLeadHours || 2;
    updateDynamicTimes(issueTime, lead, summary && summary.analysis_time_iso);

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
      // NO FABRICATED FALLBACKS (second copy of this block -- the constants
      // 28.5 / 32.0 / 80 / 16.0 / 180 / 0.0 were invented weather shown
      // whenever a field was missing). `||` also swallowed legitimate zeros,
      // so every field uses an explicit null check. Missing stays null and
      // renders as an em dash; absent rainfall is NOT a measurement of 0 mm.
      var cc = vm.currentConditions;
      cc.temperatureC = currentObs.temperature != null ? currentObs.temperature : null;
      cc.feelsLikeC = currentObs.feels_like != null ? currentObs.feels_like : null;
      cc.humidityPct = currentObs.humidity != null ? currentObs.humidity : null;
      cc.windKmh = currentObs.wind_speed != null ? currentObs.wind_speed : null;
      cc.windDirection = currentObs.wind_direction != null
        ? getWindDirection(currentObs.wind_direction) : null;
      cc.rainfallMmHr = currentObs.rainfall_1h != null ? currentObs.rainfall_1h : null;
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
        // KEY-NAME BUG: /api/nowcast/districts returns `heavy_rainfall_mm_3h`
        // and `flash_flood_pct`. This block read `rainfall_mm_3h` and
        // `flash_flood_proxy_pct`, which do not exist on that payload, so both
        // silently fell through to 0 -- the cards showed "3h Rain 0.0 mm /
        // Flash Flood 0%" beside a correct "Thunderstorm 91%" (thunderstorm
        // was the one field whose key happened to match). The backend values
        // were right all along; only this mapping was wrong.
        // Missing stays null (renders as an em dash), never a fake zero.
        var rainRaw = d.heavy_rainfall_mm_3h != null ? d.heavy_rainfall_mm_3h
          : (d.rainfall_mm_3h != null ? d.rainfall_mm_3h : null);
        var rainMm = rainRaw != null ? Number(rainRaw).toFixed(1) : null;
        var floodPct = d.flash_flood_pct != null ? d.flash_flood_pct
          : (d.flash_flood_proxy_pct != null ? d.flash_flood_proxy_pct
          : (d.flash_flood_proxy != null ? Math.round(d.flash_flood_proxy * 100) : null));
        return {
          name: name,
          district: name,
          riskLevel: level,
          overallPct: d.overall_risk_pct || Math.max(tstormPct, floodPct),
          thunderstormPct: tstormPct,
          heavyRainfallPct: rainMm,
          flashFloodPct: floodPct,
          confidencePct: d.confidence_pct != null ? d.confidence_pct : null,
          isPrimary: d.is_primary || name === "West Bengal",
          note: d.note || "StormSense multi-cell risk aggregation.",
          validUntil: d.valid_until || formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + lead * 3600 * 1000))
        };
      });
    }

    // Multi-horizon timeline points
    if (summary && summary.timeline) {
      // Pass the model's ACTUAL per-horizon predictions straight through.
      // This mapping previously substituted fixed values (28.0 C / 80% / 15 km/h)
      // for fields the forecast timeline does not carry, which is why every
      // horizon card rendered identical invented numbers.
      vm.timeline = summary.timeline.map(function (pt) {
        return {
          hours_from_now: pt.hours_from_now,
          lead_label: pt.label || ("+" + pt.hours_from_now + "h"),
          severe_weather_pct: pt.severe_weather_pct,
          rainfall_mm_3h: pt.rainfall_mm_3h,
          flash_flood_pct: pt.flash_flood_pct,
          overall_risk_pct: pt.overall_risk_pct,
          risk_level: pt.risk_level,
          stage: pt.stage,
          valid_time_formatted: pt.valid_time_formatted,
          valid_time: pt.forecast_valid_time || formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + pt.hours_from_now * 3600 * 1000))
        };
      });
    }

    if (xai) vm.explainability = xai;
    vm.systemStatus.versionLabel = "StormSense";
    vm.disclaimer = "AI-derived severe weather risk · Not an official government warning";

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
        ["kpi-ts-value", "kpi-rain-value", "kpi-ff-value", "kpi-overall-value"].forEach(id => {
            var el = document.getElementById(id);
            if (el) el.textContent = 'UNAVAILABLE';
          });
        ["kpi-ts-trend", "kpi-rain-trend", "kpi-ff-trend", "kpi-overall-trend"].forEach(id => {
            var el = document.getElementById(id);
            if (el) el.textContent = summary.message || 'Pipeline pending';
          });
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
      // This card is painted inline rather than through applyHazardCard(), so
      // its meter needs the same level-driven colouring as the other three.
      paintHazardBars("kpi-rain-bars", rHz.level, rHz.probability);
      var rainPill = document.getElementById("kpi-rain-level");
      if (rainPill) {
        var rcls = levelClass(rHz.level);
        rainPill.className = "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider "
          + rcls.bg + " " + rcls.text + " " + rcls.border;
      }
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
      // The gradient was fixed emerald->amber->orange in markup, so it ended
      // orange even at a GREEN overall level. Terminate the ramp at the level
      // actually assigned.
      paintOverallBar(overallLevel, overallProb);
      var opill = document.getElementById("kpi-overall-level");
      if (opill) {
        var ocls = levelClass(overallLevel);
        opill.className = "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider "
          + ocls.bg + " " + ocls.text + " " + ocls.border;
      }
    }

    // NOTE: the #meter-* bars are NOT written here. They live in the Current
    // Location card and are bound to the point forecast at the user's own
    // coordinates by applyPointRiskMeters(). Painting them from this state-wide
    // summary is what made the panel report another district's risk as the
    // user's local risk. The KPI cards above remain state-wide by design and
    // are labelled as such.
  }

  /**
   * Bind the Current Location risk meters to the point forecast at the user's
   * own coordinates.
   *
   * These meters live in the Current Location card, so they must describe THAT
   * location. Leaving them on the state-wide aggregate made the panel attribute
   * the statewide peak district's risk to the user's position.
   *
   * The values are FORECAST quantities for the selected horizon, which the card
   * labels explicitly -- they are never presented as current observations.
   */
  function applyPointRiskMeters(point) {
    var p = point && point.predictions;
    if (!p) {
      // No point forecast: say so rather than leaving stale statewide numbers.
      setText("meter-ts-label", "—");
      setText("meter-rain-label", "—");
      setText("meter-flood-label", "—");
      setWidth("meter-ts-bar", 0);
      setWidth("meter-rain-bar", 0);
      setWidth("meter-flood-bar", 0);
      return;
    }

    var ts = p.thunderstorm_prob_pct;
    var rain = p.heavy_rain_mm;
    var flood = p.flash_flood_proxy_pct;

    setText("meter-ts-label", ts != null ? Math.round(ts) + "%" : "—");
    setWidth("meter-ts-bar", ts != null ? ts : 0);

    setText("meter-rain-label", rain != null ? Number(rain).toFixed(1) + " mm" : "—");
    // The rain bar is a probability-style fill; scale the depth against the
    // 50 mm/3h heavy-rainfall reference so the bar stays comparable.
    setWidth("meter-rain-bar", rain != null ? Math.min(100, (Number(rain) / 50) * 100) : 0);

    setText("meter-flood-label", flood != null ? Math.round(flood) + "%" : "—");
    setWidth("meter-flood-bar", flood != null ? flood : 0);

    // Keep the horizon these numbers belong to visible next to them.
    var lead = window.currentLeadHours;
    var isHistMode = (window.stormSenseMode === "historical");
    var prefix = isHistMode ? "Case study · " : "AI forecast · ";
    // At NOW there is no forecast for t=0 -- the model's earliest horizon is
    // +2h. Say that explicitly instead of printing a bare "+2h", which reads as
    // though NOW itself were a 2-hour forecast.
    setText(
      "location-risk-horizon",
      (lead === "now" || lead == null)
        ? prefix + "earliest horizon +2h"
        : prefix + "+" + lead + "h"
    );
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
        confidencePct: d.confidence_pct != null ? d.confidence_pct : null,
        // Highest risk leads the list; no district is permanently "primary".
        isPrimary: false,
        note: d.body || "Multi-cell model risk aggregation.",
        validUntil: d.valid_until ||
          (issueTime ? formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + lead * 3600 * 1000)) : "")
      };
    });
  }

  // Severity ramp for the hazard cards' 6-bar meters.
  //
  // The segments are a SCALE, not six copies of one colour: position 1 is the
  // low end (green) and position 6 the high end (red), passing through yellow
  // and orange. So the meter reads like the legend above the dashboard --
  // green -> yellow -> orange -> red -- and how far the lit segments travel
  // along that scale is what conveys severity.
  var HAZARD_BAR_RAMP = [
    "bg-emerald-500",
    "bg-emerald-400",
    "bg-amber-400",
    "bg-amber-500",
    "bg-orange-500",
    "bg-red-500"
  ];
  var HAZARD_BAR_HEIGHTS = ["h-3", "h-4", "h-5", "h-6", "h-7", "h-9"];

  // The combined-index bar is the same green->yellow->orange->red scale as the
  // segmented meters, drawn continuously. The gradient is FIXED (it is the
  // scale itself); the filled width is what moves, so the colour at the leading
  // edge always corresponds to where the value sits on that scale.
  function paintOverallBar(level, probabilityPct) {
    var ob = document.getElementById("kpi-overall-bar");
    if (!ob) return;
    ob.className = "h-full rounded-full";
    // The element's width is the VALUE, so a gradient sized to the element
    // would squeeze the whole green->red scale into the filled part and always
    // end red. Size the gradient to the full track instead (100/pct) so each
    // colour stays pinned to its own point on the scale and the leading edge
    // shows the colour that actually corresponds to this value.
    var pct = Math.max(1, Math.min(100, Number(probabilityPct) || 0));
    ob.style.backgroundImage =
      "linear-gradient(to right, #10b981 0%, #fbbf24 40%, #f97316 70%, #ef4444 100%)";
    ob.style.backgroundSize = (100 / pct * 100) + "% 100%";
    ob.style.backgroundRepeat = "no-repeat";
  }

  function paintHazardBars(containerId, level, probabilityPct) {
    var box = document.getElementById(containerId);
    if (!box) return;
    var segs = box.children;
    // Each segment keeps its own position on the green->red scale; the
    // probability decides how far along that scale the meter is lit. Segments
    // past the value are dimmed rather than hidden so the full scale stays
    // visible and the reader can see how far the value sits along it.
    var lit = Math.max(1, Math.min(segs.length, Math.round((Number(probabilityPct) || 0) / 100 * segs.length)));
    for (var i = 0; i < segs.length; i++) {
      var on = (i < lit);
      segs[i].className = "flex-1 rounded-md " + HAZARD_BAR_HEIGHTS[i] + " "
        + (on ? HAZARD_BAR_RAMP[i] : "bg-slate-700/30");
    }
  }

  function applyHazardCard(prefix, hazard) {
    if (!hazard) return;
    var cls = levelClass(hazard.level);
    paintHazardBars(prefix + "-bars", hazard.level, hazard.probabilityPct);
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

  // -----------------------------------------------------------------------------
  // Live Dashboard Painter (Genuine Surface Obs + Honest ML Unavailable States)
  // -----------------------------------------------------------------------------
  /**
   * Paint CURRENT OBSERVED CONDITIONS only.
   *
   * This must never write the four forecast hazard cards -- those show model
   * forecast risk and are painted by paintForecastCards(). Conflating the two
   * is what previously made an observed "0.0 mm" reading appear as a heavy
   * rainfall forecast.
   */
  /**
   * Paint the HISTORICAL case study's t=0 conditions into the location panel.
   *
   * Values come from the API's surface_obs_t0 block, which the backend
   * denormalizes from the event's reanalysis input window -- real observed
   * values at the analysis time, not a forecast and not live station data.
   * A missing value renders as "—"; nothing here is defaulted or invented.
   */
  function paintHistoricalConditions(summary) {
    if (window.stormSenseMode !== "historical") return;
    // T0 only. At +2/+4/+6 the horizon painter owns these cards, otherwise the
    // event's ANALYSIS state would be displayed as a forecast for a later time.
    if (window.currentLeadHours !== "now" && window.currentLeadHours != null) {
      paintHorizonConditions();
      return;
    }
    var t0 = summary && summary.surface_obs_t0;
    function histText(v, digits, suffix) {
      if (v == null || !isFinite(Number(v))) return "—";
      return Number(v).toFixed(digits) + suffix;
    }
    if (!t0) {
      ["bind-temp", "bind-rain", "bind-humidity", "bind-wind"].forEach(function (id) {
        setText(id, "—");
      });
      ["bind-temp-source", "bind-rain-source", "bind-humidity-source", "bind-wind-source"].forEach(function (id) {
        setText(id, "Unavailable");
      });
      return;
    }
    // This painter owns the T0 heading and timestamp too. They were previously
    // written only by updateModeUI() at mode-switch time, so selecting a
    // horizon and returning to NOW left the "+6h" heading and a live valid
    // time sitting above the event's t=0 values.
    setText("location-obs-title", "Remal Historical Conditions · t=0");
    setText("location-obs-age", "analysis 26 May 2024 · 12:00 UTC");
    var t0Head = document.getElementById("location-obs-heading");
    if (t0Head) t0Head.className = "text-[11px] font-bold font-mono text-cyan-300 uppercase tracking-wider flex items-center gap-1.5";
    var t0Dot = document.getElementById("location-obs-dot");
    if (t0Dot) t0Dot.className = "size-1.5 rounded-full bg-cyan-400";
    setText("bind-temp", histText(t0.temperature_c, 1, "°C"));
    // Reanalysis rainfall at t=0 is a RATE (mm/hour), not an accumulation.
    setText("bind-rain", histText(t0.rainfall_mm, 1, " mm/h"));
    setText("bind-humidity", t0.humidity_pct != null ? t0.humidity_pct + "%" : "—");
    setText("bind-wind", histText(t0.wind_speed_kmh, 1, " km/h"));
    var histSource = "Reanalysis · 26 May 2024 12:00 UTC";
    ["bind-temp-source", "bind-humidity-source", "bind-wind-source"].forEach(function (id) {
      setText(id, histSource);
    });
    // Named as a RATE, because that is what the reanalysis field is (mm/hour),
    // not an hourly accumulation like the live station observation.
    setText("bind-rain-source", "Reanalysis rate · 26 May 2024 12:00 UTC");
  }

  /**
   * Paint the LOWER physical-conditions cards of the Current Location card so
   * they follow the SELECTED HORIZON, in both modes.
   *
   * SCIENTIFIC CONSTRAINT (why this is not "just show forecast values"):
   * the model has two output heads only -- severe-weather probability and
   * 3-hour rainfall (see predictor.predict). It does NOT predict temperature,
   * humidity, wind or pressure at any lead time. Those fields therefore exist
   * ONLY at t=0, as the analysis state that initialises the run.
   *
   * So at a forecast horizon this paints:
   *   - Rainfall  -> the genuine predicted 3 h accumulation for that horizon
   *   - T/RH/Wind -> an em dash, with the header stating they are not forecast
   * rather than silently repeating the t=0 values under a "+4h" label, which
   * would present the analysis state as a forecast.
   *
   * At NOW / T0 it defers to the observation painters, which own that state.
   */
  function paintHorizonConditions() {
    var lead = window.currentLeadHours;
    var isNow = (lead === "now" || lead === 0 || lead == null);
    var isHist = (window.stormSenseMode === "historical");

    // NOW / T0 are observation states and are painted elsewhere.
    if (isNow) return false;

    var point = window.StormSenseUserPoint;
    var summary = window.StormSenseNowcastData;

    // Prefer the forecast AT THE USER'S COORDINATES; fall back to the
    // state-wide summary only when no point forecast has been resolved, and
    // say so in the source line rather than implying it is local.
    var rainMm = null, isPointLocal = false;
    if (point && point.predictions && point.predictions.heavy_rain_mm != null) {
      rainMm = Number(point.predictions.heavy_rain_mm);
      isPointLocal = true;
    } else if (summary && summary.hazards && summary.hazards.heavy_rainfall
               && summary.hazards.heavy_rainfall.rate_mm_3h != null) {
      rainMm = Number(summary.hazards.heavy_rainfall.rate_mm_3h);
    }

    var validIso = (point && point.forecast_valid_utc)
      || (summary && summary.forecast_valid_time) || null;

    setText("location-obs-title",
      (isHist ? "Historical Model Conditions · +" : "Forecast Conditions · +") + lead + "h");
    setText("location-obs-age", validIso ? ("valid " + validIso) : ("+" + lead + "h forecast"));

    var head = document.getElementById("location-obs-heading");
    if (head) {
      head.className = "text-[11px] font-bold font-mono text-cyan-300 uppercase tracking-wider flex items-center gap-1.5";
    }
    var dot = document.getElementById("location-obs-dot");
    if (dot) dot.className = "size-1.5 rounded-full bg-cyan-400";

    // Rainfall IS forecast by the model: show it, labelled as a 3 h accumulation.
    // Only overwrite once a forecast is actually in hand. This runs once up-front
    // (to claim the heading the moment a horizon is clicked) and again when the
    // summary lands; writing "Unavailable" on that first pass would replace a
    // good reading with a false one for the whole fetch window.
    if (rainMm != null) {
      setText("bind-rain", rainMm.toFixed(1) + " mm/3h");
      setText("bind-rain-source",
        isPointLocal ? "Model forecast · 3 h accumulation · this location"
                     : "Model forecast · 3 h accumulation · state-wide peak");
    }

    // T / RH / Wind are not model outputs -- SevereWeatherNetV2 has only a severe
    // head and a rain head -- but the project already integrates the OpenWeather
    // 5-day/3-hour forecast product, which does publish them per horizon. Rainfall
    // above stays the MODEL's, these three are OpenWeather's, and the source line
    // under each card says which, so the two are never conflated.
    //
    // Historical mode is deliberately excluded: OpenWeather has no 2024 archive,
    // and today's forecast must never appear inside the Remal case study.
    if (isHist) {
      // Remal is a PAST event, so the atmosphere at +2/+4/+6h was actually
      // observed and is in the same ERA5 cache the case study reads. These are
      // verification observations (what really happened), not model output, and
      // the source line says so.
      var hl = summary && summary.surface_obs_at_lead;
      if (hl) {
        setText("bind-temp", hl.temperature_c != null ? Number(hl.temperature_c).toFixed(1) + "°C" : "—");
        setText("bind-humidity", hl.humidity_pct != null ? hl.humidity_pct + "%" : "—");
        setText("bind-wind", hl.wind_speed_kmh != null ? Number(hl.wind_speed_kmh).toFixed(1) + " km/h" : "—");
        var hsrc = "ERA5 observed · +" + lead + "h";
        setText("bind-temp-source", hsrc);
        setText("bind-humidity-source", hsrc);
        setText("bind-wind-source", hsrc);
      } else {
        ["bind-temp", "bind-humidity", "bind-wind"].forEach(function (id) {
          setText(id, "—");
        });
        ["bind-temp-source", "bind-humidity-source", "bind-wind-source"].forEach(function (id) {
          setText(id, "Observed state unavailable at this horizon");
        });
      }
      return true;
    }

    var loc = window.StormSenseUserLocation || {};
    var qlat = (loc.lat != null) ? loc.lat : 22.5726;
    var qlon = (loc.lon != null) ? loc.lon : 88.3639;

    fetch(API_BASE + "/api/forecast/conditions?lat=" + Number(qlat).toFixed(4)
          + "&lon=" + Number(qlon).toFixed(4) + "&lead=" + lead)
      .then(parseJson)
      .then(function (f) {
        // Guard against a late response landing after the user moved on.
        if (String(window.currentLeadHours) !== String(lead)) return;
        if (!f || !f.available) {
          ["bind-temp", "bind-humidity", "bind-wind"].forEach(function (id) {
            setText(id, "—");
          });
          ["bind-temp-source", "bind-humidity-source", "bind-wind-source"].forEach(function (id) {
            setText(id, (f && f.reason) ? "Forecast unavailable" : "Forecast unavailable");
          });
          return;
        }
        var src = "OpenWeather forecast · +" + lead + "h";

        setText("bind-temp", f.temperature_c != null ? Number(f.temperature_c).toFixed(1) + "°C" : "—");
        setText("bind-humidity", f.humidity_pct != null ? f.humidity_pct + "%" : "—");
        setText("bind-wind", f.wind_speed_kmh != null ? Number(f.wind_speed_kmh).toFixed(1) + " km/h" : "—");

        setText("bind-temp-source", f.temperature_c != null ? src : "Forecast unavailable");
        setText("bind-humidity-source", f.humidity_pct != null ? src : "Forecast unavailable");
        setText("bind-wind-source", f.wind_speed_kmh != null ? src : "Forecast unavailable");
      })
      .catch(function () {
        ["bind-temp-source", "bind-humidity-source", "bind-wind-source"].forEach(function (id) {
          setText(id, "Forecast unavailable");
        });
      });
    return true;
  }

  function paintDashboardLive(obs, rawLive) {
    // HARD MODE GUARD. This function paints LIVE surface observations. In the
    // historical case study there are no live observations, and writing today's
    // station readings into the panel would present current weather as part of
    // a 2024 event replay. Historical t=0 values are painted by
    // paintHistoricalConditions() instead.
    if (window.stormSenseMode === "historical") return;
    // At a forecast horizon the lower cards belong to paintHorizonConditions();
    // painting current observations there would show NOW's weather beneath a
    // "+4h" label.
    if (window.currentLeadHours !== "now" && window.currentLeadHours != null) {
      // At a forecast horizon these cards belong to paintHorizonConditions().
      // Returning WITHOUT delegating left them stuck on their "..." placeholder
      // for the whole session on first load: the dashboard opens at the default
      // +2h horizon, so this guard fired before any horizon button had been
      // clicked, and nothing else populated Temperature / Rainfall / Humidity /
      // Wind or the observation-age line. They only appeared once the user
      // happened to click a horizon. paintHistoricalConditions() already
      // delegates here for exactly this reason; the live path now matches it.
      paintHorizonConditions();
      return;
    }
    // A missing value renders as a bare "—" with NO unit appended. Rainfall in
    // particular must never default to "0.0 mm": absent data is not a
    // measurement of zero rain, and showing it as one is a false observation.
    function obsText(v, digits, suffix) {
      if (v == null || !isFinite(Number(v))) return "—";
      return Number(v).toFixed(digits) + suffix;
    }

    // Reclaim the heading. paintHorizonConditions() retitles this panel to
    // "Forecast Conditions · +Nh" and recolours it cyan; returning to NOW
    // repainted the VALUES here but left that heading in place, so live
    // observations sat under a "+2h" forecast label.
    setText("location-obs-title", "Current Observations");
    var obsHead = document.getElementById("location-obs-heading");
    if (obsHead) {
      obsHead.className = "text-[11px] font-bold font-mono text-emerald-300 uppercase tracking-wider flex items-center gap-1.5";
    }
    var obsDot = document.getElementById("location-obs-dot");
    if (obsDot) obsDot.className = "size-1.5 rounded-full bg-emerald-400";

    // Current-conditions telemetry strip
    setText("bind-temp", obsText(obs.temperature_c, 1, "°C"));
    setText("bind-rain", obsText(obs.rainfall_1h_mm, 1, " mm"));
    setText("bind-humidity", obs.humidity_pct != null ? obs.humidity_pct + "%" : "—");
    setText("bind-wind", obsText(obs.wind_speed_kmh, 1, " km/h"));

    var liveSourceLabel = rawLive && rawLive.is_stale
      ? "Surface observation · stale"
      : "Surface observation";
    setText("bind-temp-source", liveSourceLabel);
    setText("bind-rain-source", liveSourceLabel);
    setText("bind-humidity-source", liveSourceLabel);
    setText("bind-wind-source", liveSourceLabel);
      if (rawLive && rawLive.observed_at_utc) {
          var obsDate = new Date(rawLive.observed_at_utc);
          setText("nv-obs-time", obsDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
          var ageMin = Math.floor((rawLive.data_age_seconds || 0) / 60);
          setText("nv-age", ageMin + "m");
          // Same real age next to the Current Observations heading, so the
          // panel always states how old the readings beneath it are.
          setText("location-obs-age", "observed " + formatIstClock(obsDate)
            + " · " + (ageMin <= 1 ? "just now" : ageMin + " min ago"));
      } else {
          setText("nv-obs-time", "UNAVAILABLE");
          setText("nv-age", "--");
          setText("location-obs-age", "no observation");
      }


    var profLevel = document.getElementById("profile-level");
    if (profLevel) profLevel.innerHTML = '<span class="size-2 bg-emerald-400 rounded-full animate-pulse"></span> LIVE MONITORING';

    // Render Live timeline cards
    renderNowcastLive(rawLive);  // sets its own title/badge

    // Render Live High-Risk Grid notice
    renderHighRiskCellsLive();

    // Render Live Thermodynamics
    renderThermodynamicsLive(obs);

    // Render Live XAI notice
    // renderXaiLive();
  }

  // -----------------------------------------------------------------------------
  // UI Renderers for Live Mode Sections
  // -----------------------------------------------------------------------------
  function renderNowcastLive(liveObs) {
    var root = document.getElementById("nowcast-steps");
    if (!root) return;
    var obs = (liveObs && liveObs.observations) || {};

    // NEVER paint live observations while the historical case study is open.
    //
    // This strip kept whatever live values it last held when the user entered
    // historical mode, so today's Kolkata weather (26.9 C / 100% RH) sat inside
    // the Cyclone Remal replay under a "LIVE OBSERVATION" badge -- present-day
    // observations presented as part of a 2024 event. The historical painter
    // owns the strip in that mode; bail out rather than overwrite it.
    if (window.stormSenseMode === "historical") return;

    // This renderer paints OBSERVED surface values, so it owns the strip's
    // labels while it is the last writer. renderNowcast() sets the forecast
    // labels for the same container; whichever painted the cards must also be
    // the one that named them, or the badge describes the other renderer's data.
    setText("nowcast-timeline-title", "Current observed conditions");
    setText("nowcast-timeline-badge", "LIVE OBSERVATION");

    // A missing observation renders as "—", never as an invented number. These
    // fields previously fell back to fabricated readings (32.0 C, 70%, 1006 hPa,
    // 16.0 km/h, 6.5 km) that were indistinguishable from real measurements.
    var UNAVAILABLE = "—";
    function num(v, digits, suffix) {
      if (v == null || !isFinite(Number(v))) return UNAVAILABLE;
      return Number(v).toFixed(digits == null ? 1 : digits) + (suffix || "");
    }

    var feelsSub = obs.feels_like_c != null
      ? "Feels: " + num(obs.feels_like_c, 1, "°C")
      : "Feels-like unavailable";
    var desc = obs.weather_description || obs.weather || "No condition reported";

    var cards = [
      { label: "AIR TEMPERATURE", val: num(obs.temperature_c, 1, "°C"), sub: feelsSub, color: "text-white", border: "border-emerald-500/40" },
      { label: "1-HOUR RAINFALL", val: num(obs.rainfall_1h_mm, 1, " mm"), sub: "Observed precipitation", color: "text-cyan-300", border: "border-cyan-500/40" },
      { label: "RELATIVE HUMIDITY", val: obs.humidity_pct != null ? obs.humidity_pct + "%" : UNAVAILABLE, sub: "Atmospheric moisture", color: "text-blue-300", border: "border-blue-500/40" },
      { label: "SURFACE WIND", val: num(obs.wind_speed_kmh, 1, " km/h"), sub: "10 m wind", color: "text-teal-300", border: "border-teal-500/40" },
      { label: "MSL PRESSURE", val: obs.pressure_hpa != null ? obs.pressure_hpa + " hPa" : UNAVAILABLE, sub: "Barometric pressure", color: "text-indigo-300", border: "border-indigo-500/40" },
      { label: "VISIBILITY", val: num(obs.visibility_km, 1, " km"), sub: desc, color: "text-slate-200", border: "border-slate-700" }
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

  /**
   * Paint the conditions strip with the CASE STUDY's own t=0 reanalysis state.
   *
   * Counterpart to renderNowcastLive() for historical mode. Without it the
   * strip simply kept the live values it last held, so present-day weather
   * appeared inside the Cyclone Remal replay under a "LIVE OBSERVATION" badge.
   *
   * Values come from /api/nowcast/summary?mode=historical -> surface_obs_t0,
   * which the backend denormalizes from the event's ERA5 analysis window. These
   * are OBSERVED reanalysis values at the event's t=0, not a forecast and not
   * live station data, and the badge says so. Fields the reanalysis payload
   * does not carry (feels-like, visibility, sky description) render as "—"
   * rather than borrowing a live value.
   */
  function renderNowcastHistorical(summary) {
    var root = document.getElementById("nowcast-steps");
    if (!root) return;
    var t0 = summary && summary.surface_obs_t0;
    if (!t0) return;

    setText("nowcast-timeline-title", "Observed conditions at event t=0");
    setText("nowcast-timeline-badge", "ERA5 REANALYSIS");

    var UNAVAILABLE = "—";
    function num(v, digits, suffix) {
      if (v == null || !isFinite(Number(v))) return UNAVAILABLE;
      return Number(v).toFixed(digits == null ? 1 : digits) + (suffix || "");
    }
    var when = t0.time ? String(t0.time) : "event analysis time";

    var cards = [
      { label: "AIR TEMPERATURE", val: num(t0.temperature_c, 1, "°C"), sub: when, color: "text-white", border: "border-cyan-500/40" },
      { label: "RAINFALL RATE", val: num(t0.rainfall_mm, 1, " mm/h"), sub: "Reanalysis precipitation", color: "text-cyan-300", border: "border-cyan-500/40" },
      { label: "RELATIVE HUMIDITY", val: t0.humidity_pct != null ? t0.humidity_pct + "%" : UNAVAILABLE, sub: "Atmospheric moisture", color: "text-blue-300", border: "border-blue-500/40" },
      { label: "SURFACE WIND", val: num(t0.wind_speed_kmh, 1, " km/h"), sub: "10 m wind", color: "text-teal-300", border: "border-teal-500/40" },
      { label: "MSL PRESSURE", val: t0.pressure_hpa != null ? Number(t0.pressure_hpa).toFixed(1) + " hPa" : UNAVAILABLE, sub: "Barometric pressure", color: "text-indigo-300", border: "border-indigo-500/40" },
      { label: "SOURCE", val: "ERA5", sub: t0.source || "Reference atmospheric analysis", color: "text-slate-200", border: "border-slate-700" }
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

  /**
   * Paint the Atmospheric Instability State from the LIVE GFS analysis.
   *
   * Previously this function hardcoded "Unavailable (Live)" for CAPE/CIN and
   * printed the literal label "Station Anemometer" for shear -- claiming a
   * station instrument that StormSense does not read, while the live GFS
   * analysis was in fact already carrying real CAPE and CIN. It also fell back
   * to an invented 1006 hPa pressure.
   *
   * Now every field is either a real live value or an explicit unavailable
   * state naming the actual limitation. Nothing is borrowed from the historical
   * case study.
   */
  function renderThermodynamicsLive() {
    fetch(API_BASE + "/api/nowcast/thermodynamics?mode=live")
      .then(parseJson)
      .then(function (th) {
        if (!th || th.status === "unavailable") {
          setText("thermo-cape", "Unavailable");
          setText("thermo-cin", "Unavailable");
          setText("thermo-shear", "Unavailable");
          setText("thermo-wind", "Unavailable");
          setText("thermo-shear-label", "No live analysis ingested");
          setText("thermo-risk-badge", "UNAVAILABLE");
          setText("thermo-diagnostic", (th && th.message)
            || "No live GFS analysis has been ingested, so no live thermodynamic diagnostics exist.");
          setText("thermo-analysis-timestamp", "—");
          setText("thermo-source-label", "Live source unavailable");
          return;
        }

        var cape = th.cape_j_kg;
        var cin = th.cin_j_kg;
        var shear = th.bulk_shear_1000_700hpa_mps;
        var wind = th.surface_wind_kmh;

        setText("thermo-cape", cape != null ? Math.round(cape).toLocaleString() + " J/kg" : "Unavailable");
        setText("thermo-cin", cin != null ? Math.round(cin) + " J/kg" : "Unavailable");
        setText("thermo-shear", shear != null ? Number(shear).toFixed(1) + " m/s" : "Unavailable");
        setText("thermo-wind", wind != null ? Number(wind).toFixed(1) + " km/h" : "Unavailable");

        // Label the shear for the layer it was ACTUALLY computed over. Live GFS
        // ingestion carries wind at 1000/850/700 hPa, so this is not 0-6 km.
        setText("thermo-shear-label", th.bulk_shear_label || "Bulk shear");
        setText("thermo-risk-badge", th.convective_risk || "LIVE");
        setText("thermo-diagnostic", th.operational_diagnostic || "");

        // Provenance: the real analysis instant these values came from.
        setText("thermo-analysis-timestamp",
          th.analysis_time_utc ? formatUtcDateTime(parseUtcIso(th.analysis_time_utc)) : "—");
        setText("thermo-source-label", th.data_source || "NOAA GFS 0.25° f000 analysis");
      })
      .catch(function (e) {
        console.warn("Live thermodynamics fetch failed:", e);
        setText("thermo-cape", "Unavailable");
        setText("thermo-cin", "Unavailable");
        setText("thermo-shear", "Unavailable");
        setText("thermo-wind", "Unavailable");
        setText("thermo-risk-badge", "UNAVAILABLE");
        setText("thermo-diagnostic", "The thermodynamics service could not be reached. "
          + "This is a connection problem, not a statement about the atmosphere.");
        setText("thermo-analysis-timestamp", "—");
      });
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

  // -----------------------------------------------------------------------------
  // Historical Dashboard Painter
  // -----------------------------------------------------------------------------
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
        paintHazardBars("kpi-rain-bars", rHz.level, rHz.probabilityPct);
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
        paintOverallBar(ov.level, ov.probabilityPct);
        var pO = document.getElementById("kpi-overall-level");
        if (pO) {
          var clsO = levelClass(ov.level);
          pO.className = "px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider " + clsO.bg + " " + clsO.text + " " + clsO.border;
        }
      }

      // Historical meters describe the CASE-STUDY DOMAIN PEAK. That is only the
      // right thing to show when we have no user coordinates: when we do,
      // refreshLocationPanelForMode() -> applyLiveLocation() re-queries
      // /api/nowcast/point?mode=historical and paints the event's risk AT THOSE
      // COORDINATES, which is both more specific and more honest (Kolkata reads
      // 50.5% / 9.9 mm / 23.3% rather than the domain peak 100% / 47.3 mm).
      //
      // So the domain-peak fallback is painted ONLY when no location is known;
      // otherwise this would run first and be immediately overwritten, and a
      // failed point query would leave the domain peak mislabelled as local.
      var histHorizonLabel = (window.currentLeadHours === "now"
                              || window.currentLeadHours == null)
        ? "Case study · T0 (now)"
        // Previously concatenated `currentLeadHours` blindly, which rendered
        // the string "now" as "Case study · +nowh".
        : "Case study · +" + window.currentLeadHours + "h";
      setText("location-risk-horizon", histHorizonLabel);

      if (!window.StormSenseUserLocation) {
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
      }

      var profLevel = document.getElementById("profile-level");
      if (profLevel) profLevel.innerHTML = '<span class="size-2 bg-cyan-400 rounded-full animate-pulse"></span> HISTORICAL CASE STUDY';
    }

    // In historical mode the location panel shows the event's t=0 reanalysis
    // conditions. This runs BEFORE the live telemetry strip below, which is
    // skipped in historical mode so it cannot overwrite these values with
    // live-shaped data (including a wind DIRECTION the t=0 payload lacks).
    paintHistoricalConditions(data.rawSummary || window.StormSenseNowcastData);
    paintHorizonConditions();

    // Telemetry strip -- live mode only.
    if (data.currentConditions && window.stormSenseMode !== "historical"
        && (window.currentLeadHours === "now" || window.currentLeadHours == null)) {
      setText("bind-temp", (data.currentConditions.temperatureC != null ? data.currentConditions.temperatureC.toFixed(1) + "°C" : "—"));
      setText("bind-rain", (data.currentConditions.rainfallMmHr != null ? Number(data.currentConditions.rainfallMmHr).toFixed(1) + " mm" : "—"));
      setText("bind-humidity", (data.currentConditions.humidityPct != null ? data.currentConditions.humidityPct + "%" : "—"));
      setText("bind-wind", (data.currentConditions.windKmh != null ? data.currentConditions.windKmh.toFixed(1) + " km/h " + (data.currentConditions.windDirection || "") : "—"));

      var sourceLabel = "HISTORICAL INPUT (t=0) · 26 May 2024 12:00 UTC";
      setText("bind-temp-source", sourceLabel);
      setText("bind-rain-source", sourceLabel);
      setText("bind-humidity-source", sourceLabel);
      setText("bind-wind-source", sourceLabel);
    }

    // Render Historical timeline
    if (data.timeline && data.timeline.length) {
      renderNowcast(data.timeline);  // sets its own title/badge
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

  // Threshold above which an area is listed as a currently identified
  // higher-risk area. Matches the backend's ACTIVE_RISK_THRESHOLD_PCT so the
  // bulletin list and the map are driven by the same forecast state.
  var BULLETIN_MIN_OVERALL_PCT = 25;

  /**
   * Active Meteorological Bulletins: the currently identified higher-risk
   * districts, built from the SAME district aggregation of the SAME forecast
   * grid that paints the map. Nothing here is hardcoded -- if no district is
   * above the watch level, the panel says so rather than inventing entries.
   */
  window.renderBulletinsFromDistricts = function (districts, summary) {
    var root = document.getElementById("active-bulletins-list");
    var countEl = document.getElementById("bulletins-count");
    var subEl = document.getElementById("bulletins-subtitle");
    if (!root) return;

    var lead = apiLeadHours();
    var horizon = "+" + lead + "h";

    if (!districts || !districts.length) {
      root.innerHTML = '<div class="p-4 text-xs text-slate-400 font-mono col-span-full">'
        + 'District advisories are unavailable for this forecast state.</div>';
      if (countEl) countEl.textContent = "";
      return;
    }

    var active = districts.filter(function (d) {
      var ov = d.overall_pct != null ? d.overall_pct : 0;
      return ov >= BULLETIN_MIN_OVERALL_PCT || d.alert === true;
    });

    if (subEl) {
      subEl.textContent = "Currently identified higher-risk areas · " + horizon + " forecast";
    }

    if (!active.length) {
      root.innerHTML = '<div class="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/25 '
        + 'text-xs text-emerald-300 font-mono col-span-full">'
        + 'No district currently exceeds the watch level at ' + horizon + '. '
        + 'All monitored districts are in normal conditions.</div>';
      if (countEl) countEl.textContent = "0 areas";
      return;
    }

    if (countEl) {
      countEl.textContent = active.length + (active.length === 1 ? " area" : " areas");
    }

    root.innerHTML = active.map(function (d) {
      var cls = levelClass(d.risk_level || "green");
      var name = d.district || "District";
      var validTime = d.valid_until || "";
      var ts = d.thunderstorm_pct != null ? d.thunderstorm_pct : 0;
      var rain = d.heavy_rainfall_mm_3h != null ? d.heavy_rainfall_mm_3h : 0;
      var flood = d.flash_flood_pct != null ? d.flash_flood_pct : 0;
      var stage = d.stage || "NORMAL";

      // Risk wording derives from the model's own stage, not a fixed string.
      var riskWord = stage === "WARNING" ? "High"
        : stage === "ALERT" ? "Elevated"
        : stage === "WATCH" ? "Moderate" : "Normal";

      return (
        '<div class="p-4 rounded-xl ' + cls.bg + ' border ' + cls.border + ' flex flex-col gap-2 cursor-pointer hover:brightness-125 transition-all" '
          + 'onclick="window.focusBulletinArea(\'' + String(name).replace(/'/g, "\\'") + '\')" '
          + 'title="Show ' + name + ' on the map">' +
          '<div class="flex items-start justify-between gap-2">' +
            '<h4 class="text-sm font-bold text-white leading-tight">' + name + '</h4>' +
            '<span class="px-2 py-0.5 rounded text-[10px] font-extrabold uppercase font-mono tracking-wider shrink-0 ' + cls.pill + '">' + horizon + '</span>' +
          '</div>' +
          '<div class="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] font-mono">' +
            '<span class="text-slate-400">Thunderstorm</span>'
              + '<strong class="text-white text-right">' + Math.round(ts) + '%</strong>' +
            '<span class="text-slate-400">Heavy rainfall</span>'
              + '<strong class="text-white text-right">' + Number(rain).toFixed(1) + ' mm</strong>' +
            '<span class="text-slate-400">Flash-flood proxy</span>'
              + '<strong class="text-white text-right">' + Math.round(flood) + '%</strong>' +
            '<span class="text-slate-400">Risk</span>'
              + '<strong class="' + cls.text + ' text-right">' + riskWord + '</strong>' +
          '</div>' +
          '<div class="text-[10px] text-slate-400 leading-relaxed border-t border-slate-700/40 pt-1.5">'
            + (d.body || "") + '</div>' +
          (validTime
            ? '<div class="text-[10px] text-slate-500 font-mono">Valid: ' + validTime + '</div>'
            : '') +
        '</div>'
      );
    }).join("");
  };

  // Clicking a bulletin focuses the map on that district, using the real
  // boundary geometry from /api/geo/areas.
  window.focusBulletinArea = function (districtName) {
    var areas = window.StormSenseAreas || [];
    for (var i = 0; i < areas.length; i++) {
      if (areas[i].name === districtName) {
        window.jumpToArea(areas[i].id);
        var sel = document.getElementById("sector-selector");
        if (sel) sel.value = areas[i].id;
        return true;
      }
    }
    window.showToast("No map geometry available for " + districtName, "warning");
    return false;
  };

  function renderNowcast(timeline) {
    var root = document.getElementById("nowcast-steps");
    if (!root || !timeline || !timeline.length) return;

    // OWNERSHIP OF #nowcast-steps.
    //
    // This strip is the "Current observed conditions" panel. renderNowcast()
    // (the 0-6 hour forecast row) and renderNowcastLive() (observed conditions)
    // both painted into the SAME container, so whichever ran last won and the
    // panel flipped between the two depending on load order and which horizon
    // was clicked.
    //
    // Per the product decision the strip always shows CURRENT OBSERVED
    // CONDITIONS, so the forecast row no longer writes here. Nothing is lost:
    // the 0-6 hour forecast remains fully available through the horizon
    // buttons, the risk map, the hazard cards, the popup and the legend --
    // this function simply stops hijacking the observation strip.
    //
    // The timeline payload is still kept on window for any consumer that wants
    // it, so no data is discarded.
    window.StormSenseTimeline = timeline;

    // Repaint the strip with CURRENT OBSERVED CONDITIONS instead. Without this
    // the strip was simply left as-is when a forecast horizon was selected,
    // so it kept the stale "0-6 hour outlook" heading with no cards under it.
    // The observed values are the same in every horizon (they are observations,
    // not forecasts); only the map, hazard cards and popup follow the horizon.
    if (window.stormSenseMode === "live" && window.StormSenseLiveSurface
        && typeof renderNowcastLive === "function") {
      renderNowcastLive(window.StormSenseLiveSurface);
    }
    return;
  }

  /** Retained for reference; not wired to #nowcast-steps any more. */
  function renderNowcastForecastRow(timeline) {
    var root = document.getElementById("nowcast-steps");
    if (!root || !timeline || !timeline.length) return;
    var activeLead = window.currentLeadHours || 2;

    // These cards are MODEL PREDICTIONS carrying future valid times. The badge
    // is set here, in the one function that paints them, because the live
    // refresh path calls this directly and used to leave the strip labelled
    // "LIVE OBSERVATION" from paintDashboardLive() -- forecast values sitting
    // under a live-observation badge.
    setText("nowcast-timeline-title", "0–6 hour forecast outlook");
    setText("nowcast-timeline-badge",
      window.stormSenseMode === "historical" ? "HISTORICAL FORCING" : "AI FORECAST");

    root.innerHTML = timeline.map(function (point, index) {
      var isTargetLead = Number(point.hours_from_now) === Number(activeLead);
      var borderCls = isTargetLead ? "border-cyan-500 bg-cyan-950/30 ring-1 ring-cyan-500/50" : "border-slate-800 bg-slate-950/80";
      var label = "+" + point.hours_from_now + "h";

      // These cards show what the model ACTUALLY predicts at each horizon:
      // severe-weather probability, 3h rainfall and the flash-flood proxy.
      //
      // They previously displayed temperature/humidity/wind with hardcoded
      // fallbacks (28.0 C / 80% / 15 km/h) because the forecast timeline
      // carries no such fields -- so every horizon rendered the same invented
      // numbers. The model has no ambient-weather forecast head, so those rows
      // are gone rather than faked.
      var sev = point.severe_weather_pct;
      var rain = point.rainfall_mm_3h;
      var flood = point.flash_flood_pct;
      var validTime = point.valid_time_formatted || "";

      function val(v, suffix, dp) {
        if (v == null || isNaN(Number(v))) return '<span class="text-slate-500">n/a</span>';
        return Number(v).toFixed(dp == null ? 0 : dp) + suffix;
      }

      // Each card states BOTH halves of the temporal contract: the exact valid
      // instant in IST, and the exact offset from NOW. The valid instant is
      // derived from the same authoritative reference the API issued, not from
      // the browser clock, so card and map can never disagree.
      var validIso = point.forecast_valid_time || point.valid_time || point.valid_time_utc;
      var validIst = validIso ? formatIstStamp(parseUtcIso(validIso)) : "";
      var offsetLabel = "+" + point.hours_from_now + ":00 from now";

      return (
        '<div class="p-3 rounded-xl border text-center transition-all cursor-pointer ' + borderCls + '" onclick="window.setForecastHorizon(null, null, ' + point.hours_from_now + ')" title="Valid ' + validTime + '">' +
          '<div class="text-[10px] text-slate-400 font-mono uppercase font-bold">' + label + '</div>' +
          '<div class="text-white font-bold font-mono text-base mt-1">' + val(sev, "%") + '</div>' +
          '<div class="text-[9px] text-slate-500 font-mono uppercase">severe risk</div>' +
          '<div class="text-[10px] text-cyan-400 mt-1.5 font-mono">' + val(rain, " mm/3h", 1) + '</div>' +
          '<div class="text-[10px] text-amber-300 mt-1 font-mono">Flood proxy ' + val(flood, "%") + '</div>' +
          (validIst
            ? '<div class="text-[9px] text-white font-mono font-bold mt-1.5 pt-1.5 border-t border-slate-800">' + validIst + '</div>'
            : '') +
          '<div class="text-[9px] text-amber-300/90 font-mono">' + offsetLabel + '</div>' +
          '<div class="text-[8px] text-slate-600 mt-0.5 truncate font-mono">' + validTime + ' UTC</div>' +
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
      var rainVal = d.heavyRainfallPct != null ? d.heavyRainfallPct
        : (d.heavy_rainfall_mm_3h != null ? d.heavy_rainfall_mm_3h
        : (d.rainfall_mm != null ? d.rainfall_mm : null));
      var floodPct = d.flashFloodPct != null ? d.flashFloodPct
        : (d.flash_flood_pct != null ? d.flash_flood_pct : null);
      var overallPct = d.overallPct != null ? d.overallPct : (d.overall_pct != null ? d.overall_pct : 0);
      var note = d.note || d.body || "StormSense multi-cell risk aggregation across district boundaries.";
      var isPrimary = d.isPrimary != null ? d.isPrimary : (d.is_primary || name === "West Bengal");
      var cls = levelClass(riskLevel);
      var ring = isPrimary ? "border-2 border-cyan-500/80 shadow-cyan-500/10" : "border " + cls.border;
      // Only shown when the backend actually supplies it -- never defaulted to
      // an invented "92%".
      var conf = d.confidencePct != null ? d.confidencePct : (d.confidence_pct != null ? d.confidence_pct : null);
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
              '<div><span class="text-[10px] text-slate-400 block">3h RAIN</span><span class="text-cyan-400 font-bold text-base">' + (rainVal != null ? Number(rainVal).toFixed(1) + ' mm' : '—') + '</span></div>' +
              '<div><span class="text-[10px] text-slate-400 block">FLASH FLOOD</span><span class="text-amber-400 font-bold text-base">' + (floodPct != null ? floodPct + '%' : '—') + '</span></div>' +
            '</div>' +
          '</div>' +
          '<div class="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">' +
            '<span>Peak Risk: <strong class="text-white">' + overallPct + '%</strong></span>' +
            // "Model Skill" was previously a hardcoded 92% presented as if it
            // were a measured per-district figure. It is only shown when the
            // backend genuinely supplies one.
            (conf != null
              ? '<span>Confidence: <strong class="text-emerald-400">' + conf + '%</strong></span>'
              : '') +
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

  /**
   * Paint the Atmospheric Instability State from a thermodynamics payload.
   *
   * Handles both shapes explicitly. A LIVE payload reports shear over the
   * 1000-700 hPa layer that live GFS ingestion actually carries and leaves
   * `bulk_shear_0_6km_mps` null; a HISTORICAL payload reports a genuine 0-6 km
   * shear from the ERA5 profile. Reading the wrong field for the wrong mode is
   * what would silently blank live values or mislabel their vertical extent.
   */
  function renderThermodynamics(thermo) {
    if (!thermo) return;

    if (thermo.status === "unavailable") {
      setText("thermo-cape", "Unavailable");
      setText("thermo-cin", "Unavailable");
      setText("thermo-shear", "Unavailable");
      setText("thermo-wind", "Unavailable");
      setText("thermo-shear-label", thermo.mode === "live"
        ? "No live analysis ingested" : "Unavailable");
      setText("thermo-risk-badge", "UNAVAILABLE");
      setText("thermo-diagnostic", thermo.message
        || "Thermodynamic diagnostics are unavailable for this mode.");
      setText("thermo-analysis-timestamp", "—");
      setText("thermo-source-label", "Unavailable");
      return;
    }

    var isLive = thermo.mode === "live" || thermo.source === "live_gfs_analysis";

    var cape = thermo.cape_surface != null ? thermo.cape_surface : thermo.cape_j_kg;
    setText("thermo-cape", cape != null ? Math.round(cape).toLocaleString() + " J/kg" : "Unavailable");

    var cin = thermo.cin != null ? thermo.cin : thermo.cin_j_kg;
    setText("thermo-cin", cin != null ? Math.round(cin) + " J/kg" : "Unavailable");

    // Pick the shear field that matches the payload's own vertical extent.
    var shear = isLive
      ? thermo.bulk_shear_1000_700hpa_mps
      : (thermo.bulk_shear_0_6km_ms != null ? thermo.bulk_shear_0_6km_ms : thermo.bulk_shear_0_6km_mps);
    setText("thermo-shear", shear != null ? Number(shear).toFixed(1) + " m/s" : "Unavailable");

    if (isLive) {
      var wind = thermo.surface_wind_kmh;
      setText("thermo-wind", wind != null ? Number(wind).toFixed(1) + " km/h" : "Unavailable");
      setText("thermo-shear-label", thermo.bulk_shear_label || "Bulk shear");
      setText("thermo-source-label", thermo.data_source || "NOAA GFS 0.25° f000 analysis");
      setText("thermo-analysis-timestamp",
        thermo.analysis_time_utc ? formatUtcDateTime(parseUtcIso(thermo.analysis_time_utc)) : "—");
    } else {
      var hWind = thermo.wind_speed_kmh;
      setText("thermo-wind", hWind != null ? Number(hWind).toFixed(1) + " km/h" : "Unavailable");
      setText("thermo-shear-label", thermo.wind_shear_interpretation || "0–6 km bulk shear");
      setText("thermo-source-label", thermo.data_source || "Historical case-study profile");
      setText("thermo-analysis-timestamp", thermo.issue_time || "—");
    }

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

    // PEAK = the single highest-probability GRID CELL at this horizon (cells are
    // returned sorted descending). It is a one-cell maximum over 825 cells, NOT
    // a state-wide or district risk, so it is labelled as a cell maximum and
    // carries the cell's location -- "PEAK PROB: 100%" alone reads as though
    // the whole state were certain.
    var topProb = Math.round(Number(cells[0].thunderstorm_prob || 0) * 1000) / 10;
    var peakBadge = document.getElementById("desk-peak-prob-badge");
    if (peakBadge) {
      var pc = cells[0];
      var where = (pc.lat != null && pc.lon != null)
        ? " @ " + Number(pc.lat).toFixed(2) + "N," + Number(pc.lon).toFixed(2) + "E"
        : "";
      // In historical mode this badge sits over the OBSERVED t=0 rainfall field,
      // so a forecast probability would describe a different quantity than the
      // map beneath it. Report the observed domain peak instead.
      var histObs = window.StormSenseHistoricalAnalysisState;
      if (window.stormSenseMode === "historical" && histObs && histObs.domain_max_mm_h != null) {
        peakBadge.textContent = "OBSERVED PEAK: " + Number(histObs.domain_max_mm_h).toFixed(1) + " mm/h";
        peakBadge.title = "Highest observed ERA5 rainfall rate in the domain at the "
          + "case study's analysis time (t=0) — an observation, not a forecast.";
      } else {
        peakBadge.textContent = "PEAK CELL: " + topProb.toFixed(1) + "%" + where;
        peakBadge.title = "Highest single 0.25° grid cell probability at this horizon "
          + "(maximum over 825 cells) — not a district or state-wide value.";
      }
    }

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

  // -----------------------------------------------------------------------------
  // Model Benchmarks Desk (View 4)
  // -----------------------------------------------------------------------------
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

    // Every label that DESCRIBES the selected horizon is bound here, because
    // this is the one function that all selection paths (initial load and
    // setBenchmarkLead) go through. Previously these two elements kept their
    // hardcoded "+2h" markup while the metrics beneath them changed, so the
    // desk actively misreported which horizon the numbers belonged to.
    // The static +2h..+6h verification TABLE is left alone: each of its rows
    // genuinely describes its own horizon and must not follow the selection.
    setText("benchmark-selected-lead-label", "+" + lead + "h Horizon (" + (lead * 60) + "-minute lead)");
    setText("benchmark-lead-badge", "Lead Time: +" + lead + " Hours");

    // Update AI Forecast Card
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
    // The API carries no PER-HORIZON Brier for the baselines. When it is absent,
    // leave the markup's documented across-horizon mean in place rather than
    // blanking it -- and never redraw it as if it tracked the selected horizon.
    if (v1.brier_score != null) setText("bm-v1-brier", Number(v1.brier_score).toFixed(4));
    setText("bm-v1-mae", ((v1.mae != null ? v1.mae : v1.rainfall_mae_mm) != null ? Number(v1.mae != null ? v1.mae : v1.rainfall_mae_mm).toFixed(2) + " mm" : "—"));

    // Update Persistence Card
    setText("bm-pers-csi", (pers.csi != null ? Number(pers.csi).toFixed(4) : "—"));
    setText("bm-pers-prauc", (pers.pr_auc != null ? Number(pers.pr_auc).toFixed(4) : "—"));
    setText("bm-pers-pod", ((pers.pod != null ? pers.pod : pers.recall_pod) != null ? Number(pers.pod != null ? pers.pod : pers.recall_pod).toFixed(4) : "—"));
    setText("bm-pers-far", (pers.far != null ? Number(pers.far).toFixed(4) : "—"));
    if (pers.brier_score != null) setText("bm-pers-brier", Number(pers.brier_score).toFixed(4));
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
              '<div class="flex justify-between"><span>AI Forecast CSI:</span><strong class="text-emerald-400">' + v2Csi + '</strong></div>' +
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

  // -----------------------------------------------------------------------------
  // Leaflet Geospatial Rendering Engine (Zero Watermarks, Public Free Dark Tiles)
  // -----------------------------------------------------------------------------
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

    // INTERACTION: the district layer is drawn but NOT interactive.
    //
    // It used to carry fillOpacity 0.02 -- visually almost nothing, but a solid
    // clickable polygon covering the whole state. Google routed every pointer
    // event over West Bengal to it, so scroll-wheel zoom-in did nothing and
    // clicks opened the district name tooltip instead of the risk analysis. The
    // fill is kept for its faint tint, but the layer no longer captures events.
    //
    // District identification is not lost: handleMapClick() sends the clicked
    // coordinate to /api/nowcast/point, and the BACKEND resolves which district
    // and which 0.25-degree model cell contains it (see the "district" and
    // "grid_cell" fields). That is a spatial lookup against the real boundary
    // dataset rather than a DOM hit-test, so it stays correct at every zoom.
    var boundaryLayer = L.geoJSON(geojsonData, {
      style: {
        color: "#38bdf8",
        weight: 1.0,
        opacity: 0.65,
        fillColor: "#0284c7",
        fillOpacity: 0.02
      },
      interactive: false
    });

    boundaryLayer.addTo(mapInstance);
    mapInstance._boundaryLayer = boundaryLayer;
  }

  function removeRiskSurface(mapInstance) {
    if (mapInstance && mapInstance._riskSurfaceOverlay) {
      mapInstance.removeLayer(mapInstance._riskSurfaceOverlay);
      mapInstance._riskSurfaceOverlay = null;
    }
  }

  // ---------------------------------------------------------------------------
  // CURRENT-OBSERVATION SURFACE (the NOW layer)
  //
  // Source: /api/observations/surface -- current station readings from the
  // project's existing current-weather provider, interpolated between stations
  // and left blank where no station is near enough. It contains NO model output
  // and NO forecast horizon, and is drawn in a blue/teal ramp that is
  // deliberately distinct from the forecast risk ramp.
  // ---------------------------------------------------------------------------
  window.StormSenseObsVariable = "rain_1h_mm";

  function removeObservationSurface(mapInstance) {
    if (mapInstance && mapInstance._observationOverlay) {
      mapInstance.removeLayer(mapInstance._observationOverlay);
      mapInstance._observationOverlay = null;
    }
    if (mapInstance && mapInstance._observationStationLayer) {
      mapInstance.removeLayer(mapInstance._observationStationLayer);
      mapInstance._observationStationLayer = null;
    }
  }

  function renderObservationSurface(mapInstance) {
    if (!mapInstance || typeof L === "undefined") return;

    // Historical mode is a frozen case study and has no live observations.
    if (window.stormSenseMode === "historical") {
      removeObservationSurface(mapInstance);
      return;
    }

    var variable = window.StormSenseObsVariable || "rain_1h_mm";
    var url = API_BASE + "/api/observations/surface?variable=" + variable
      + "&t=" + Date.now();

    if (mapInstance._observationOverlay) {
      mapInstance._observationOverlay.setUrl(url);
    } else {
      var overlay = L.imageOverlay(url, window.WB_BOUNDS, {
        opacity: 0.9,
        interactive: false,
        // Below the risk surface's z-index; they are never shown together, but
        // the ordering keeps the station markers on top of the field.
        zIndex: 290
      });
      overlay.addTo(mapInstance);
      mapInstance._observationOverlay = overlay;
    }

    // Draw the actual reporting stations. Showing WHERE the real measurements
    // are is what makes the interpolation between them honest rather than an
    // undifferentiated wash of colour.
    fetch(API_BASE + "/api/observations/current?variable=" + variable)
      .then(parseJson)
      .then(function (d) {
        if (!d || d.status !== "ok") {
          updateObservationProvenance(null, d);
          return;
        }
        window.StormSenseObservations = d;
        updateObservationProvenance(d, null);
        drawObservationStations(mapInstance, d);
      })
      .catch(function (e) {
        console.warn("Observation surface metadata unavailable:", e);
        updateObservationProvenance(null, null);
      });
  }

  function drawObservationStations(mapInstance, data) {
    if (!mapInstance || !data || !data.stations) return;
    if (mapInstance._observationStationLayer) {
      mapInstance.removeLayer(mapInstance._observationStationLayer);
      mapInstance._observationStationLayer = null;
    }
    if (window.currentLeadHours !== "now") return;

    var group = L.layerGroup();
    var unit = data.unit || "";
    data.stations.forEach(function (s) {
      if (s.lat == null || s.lon == null) return;
      var val = s[data.variable];
      // Neutral slate, not blue: the rainfall field is now a yellow->red ramp,
      // and a blue dot on it reads as another data value rather than as a
      // location marker for where the measurements were actually taken.
      var marker = L.circleMarker([s.lat, s.lon], {
        radius: 3.5,
        color: "#f8fafc",
        weight: 1,
        opacity: 0.95,
        fillColor: "#334155",
        fillOpacity: 0.9
      });

      var rows =
        '<div style="display:flex;justify-content:space-between;gap:10px;"><span>Temperature:</span><strong>' +
          (s.temperature_c != null ? Number(s.temperature_c).toFixed(1) + "°C" : "—") + '</strong></div>' +
        '<div style="display:flex;justify-content:space-between;gap:10px;"><span>Rain (1h):</span><strong>' +
          (s.rain_1h_mm != null ? Number(s.rain_1h_mm).toFixed(2) + " mm" : "—") + '</strong></div>' +
        '<div style="display:flex;justify-content:space-between;gap:10px;"><span>Humidity:</span><strong>' +
          (s.humidity_pct != null ? s.humidity_pct + "%" : "—") + '</strong></div>' +
        '<div style="display:flex;justify-content:space-between;gap:10px;"><span>Cloud:</span><strong>' +
          (s.cloud_pct != null ? s.cloud_pct + "%" : "—") + '</strong></div>' +
        '<div style="display:flex;justify-content:space-between;gap:10px;"><span>Wind:</span><strong>' +
          (s.wind_kmh != null ? Number(s.wind_kmh).toFixed(1) + " km/h" : "—") + '</strong></div>';

      marker.bindPopup(
        '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;' +
          'background:#0f172a;border-radius:10px;min-width:210px;border:1px solid #0369a1;">' +
          '<div style="font-weight:800;font-size:12px;margin-bottom:4px;color:#38bdf8;">' +
            'OBSERVING STATION</div>' +
          '<div style="color:#ffffff;font-weight:bold;margin-bottom:4px;">' +
            (s.name || "Station") + '</div>' +
          rows +
          '<div style="font-size:9px;color:#64748b;border-top:1px solid #1e293b;' +
            'padding-top:4px;margin-top:5px;">Measured here · ' +
            (s.observed_at_utc ? formatUtcDateTime(parseUtcIso(s.observed_at_utc)) : "time unknown") +
          '</div>' +
        '</div>'
      );
      group.addLayer(marker);
    });

    group.addTo(mapInstance);
    mapInstance._observationStationLayer = group;
  }

  // Writes the NOW provenance line: what was measured, when, and the explicit
  // statement that between-station values are interpolated.
  function updateObservationProvenance(data, errPayload) {
    var el = document.getElementById("obs-provenance");
    if (!el) return;

    if (!data) {
      el.textContent = (errPayload && errPayload.message)
        || "Current observations unavailable.";
      return;
    }
    var ageMin = data.max_observation_age_seconds != null
      ? Math.round(data.max_observation_age_seconds / 60) : null;
    el.textContent =
      data.stations_reporting + "/" + data.stations_requested + " stations · "
      + (data.variable_label || "") + " "
      + (data.value_range && data.value_range.max != null
          ? "up to " + data.value_range.max + " " + (data.unit || "") : "")
      + (ageMin != null ? " · observed ≤" + ageMin + " min ago" : "")
      + " · interpolated between stations";
  }

  function removeHistoricalAnalysisSurface(mapInstance) {
    if (mapInstance && mapInstance._historicalAnalysisOverlay) {
      mapInstance.removeLayer(mapInstance._historicalAnalysisOverlay);
      mapInstance._historicalAnalysisOverlay = null;
    }
  }

  // Historical case study t=0 field: the OBSERVED reanalysis state at the
  // event's analysis time, rendered in the blue/teal observation palette so it
  // is visually distinct from the green->red forecast risk ramp.
  function renderHistoricalAnalysisSurface(mapInstance) {
    if (!mapInstance || typeof L === "undefined") return;
    if (window.stormSenseMode !== "historical") {
      removeHistoricalAnalysisSurface(mapInstance);
      return;
    }
    // Provenance for the field being drawn (observed peak, units, source). The
    // desk badge must report the same quantity as the overlay, so repaint it
    // once this resolves rather than leaving the forecast percentage up.
    if (!window.StormSenseHistoricalAnalysisState) {
      fetch(API_BASE + "/api/historical/analysis-state").then(parseJson)
        .then(function (st) {
          if (!st || st.status !== "ok") return;
          window.StormSenseHistoricalAnalysisState = st;
          var pb = document.getElementById("desk-peak-prob-badge");
          if (pb && window.stormSenseMode === "historical" && st.domain_max_mm_h != null) {
            pb.textContent = "OBSERVED PEAK: " + Number(st.domain_max_mm_h).toFixed(1) + " mm/h";
            pb.title = "Highest observed ERA5 rainfall rate in the domain at the "
              + "case study's analysis time (t=0) — an observation, not a forecast.";
          }
        }).catch(function () { /* badge falls back to the forecast label */ });
    }
    var url = API_BASE + "/api/historical/analysis-surface?variable=rain_mm";
    if (mapInstance._historicalAnalysisOverlay) {
      mapInstance._historicalAnalysisOverlay.setUrl(url);
    } else {
      var overlay = L.imageOverlay(url, window.WB_BOUNDS, {
        opacity: 1.0,
        interactive: false,
        zIndex: 290
      });
      overlay.addTo(mapInstance);
      mapInstance._historicalAnalysisOverlay = overlay;
    }
  }

  function renderContinuousRiskSurface(mapInstance, leadHours) {
    if (!mapInstance || typeof L === "undefined") return;

    // NOW IS NOT A FORECAST -- but it IS a StormSense spatial risk field.
    //
    // History of this branch matters, because there are two distinct mistakes
    // to avoid and only one of them was previously addressed:
    //
    //   1. NEVER relabel +2h as NOW. This function once coerced 'now' to lead=2
    //      via apiLeadHours(), painting the +2h PREDICTED field under a legend
    //      reading "CURRENT OBSERVATIONS" -- future model output presented as
    //      the present. That is still forbidden and is NOT what happens below.
    //
    //   2. NEVER leave NOW blank. The fix for (1) swapped NOW to a live
    //      station-observation layer only. Live surface coverage over West
    //      Bengal is sparse, so that layer renders very nearly empty and NOW
    //      showed no spatial risk at all, while +2/+4/+6 showed full fields.
    //
    // NOW therefore paints the model's OWN T=0 risk field: lead=0, which the
    // service builds from a genuinely separate inference (the T-2h analysis
    // advanced to T0), not a copy of the +2h slice. It is verified distinct
    // from lead=2 at the API. The live observation layer remains a SEPARATE
    // product and is drawn as station markers on top, so model risk and real
    // observations stay visually and semantically distinct.
    if (leadHours === 'now') {
      // CHANGE 6: the Interactive Nowcasting Map shows the STORMSENSE MODEL
      // RISK field in BOTH modes, at every horizon including NOW/T0.
      //
      // Historical NOW previously painted the ERA5 OBSERVED RAINFALL analysis
      // here (a blue/cyan mm/h field). That is a legitimate product, but it is
      // a different quantity from model risk, so the case study's risk map did
      // not look or behave like the risk map everywhere else on the site.
      //
      // The backend serves a real historical lead=0 risk surface -- a separate
      // inference, verified byte-distinct from lead=2, not a relabelled +2h --
      // so historical NOW now uses the same risk painter as live NOW and as
      // +2/+4/+6. The observed rainfall field remains available as its own
      // clearly-labelled product on the Radar & Satellite Feeds view
      // (initRadarMap/renderHistoricalAnalysisSurface), which is where an
      // observation belongs. The two are never drawn on top of each other.
      removeHistoricalAnalysisSurface(mapInstance);
      removeObservationSurface(mapInstance);
      // Fall through to the shared risk-surface painter with the model's real
      // lead=0 field, so NOW is visually consistent with the forecast horizons.
    }

    // Any forecast horizon drops the historical t=0 analysis layer, so an
    // observed field never sits underneath a prediction.
    removeHistoricalAnalysisSurface(mapInstance);

    var isNowHorizon = (leadHours === 'now');

    // Any FORECAST horizon drops the observation layer, so current data never
    // sits underneath a prediction. NOW is not a forecast: there the observed
    // field is a legitimate companion to the T=0 risk surface, and the two are
    // labelled separately, so it is kept.
    if (!isNowHorizon) {
      removeObservationSurface(mapInstance);
    }

    // 'now' is the model's lead=0 head. apiLeadHours() already maps it to 0;
    // resolving it here too keeps the string out of the query parameter, which
    // would otherwise request risk-surface?lead=now and 422.
    var lead = isNowHorizon
      ? 0
      : ((leadHours == null) ? apiLeadHours() : leadHours);
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

  // The AI forecast is communicated by the spatial risk FIELD (the 0.25 degree
  // model grid rendered as a continuous surface), not by scattered marker dots.
  // The previous "PREDICTED HIGH-RISK ML CELL #n" beacons made a continuous
  // hazard field look like a handful of mysterious points and are removed.
  // This function now only clears any previously drawn beacons.
  function renderHotspotBeacons(mapInstance) {
    if (!mapInstance || typeof L === "undefined") return;
    if (mapInstance._hotspotLayer) {
      mapInstance.removeLayer(mapInstance._hotspotLayer);
      mapInstance._hotspotLayer = null;
    }
  }

  // ---------------------------------------------------------------------------
  // Live location marker (violet #a855f7 -- deliberately outside the risk ramp,
  // so it can never be read as a severity level, and distinct from the cyan used
  // for AI-forecast accents). Uses the real detected coordinates; when the
  // location is unknown no marker is drawn and the UI says so rather than
  // pretending a target-region coordinate is the user's position.
  // ---------------------------------------------------------------------------
  window.StormSenseUserLocation = null;

  function removeLiveLocationMarker(mapInstance) {
    if (mapInstance && mapInstance._liveLocationMarker) {
      mapInstance.removeLayer(mapInstance._liveLocationMarker);
      mapInstance._liveLocationMarker = null;
    }
  }

  function renderLiveLocationMarker(mapInstance) {
    if (!mapInstance || typeof L === "undefined") return;
    removeLiveLocationMarker(mapInstance);

    // The marker is drawn in BOTH modes.
    //
    // It was previously suppressed in historical mode on the grounds that a
    // live position "has no meaning inside a frozen 2024 event replay". That
    // reasoning does not hold: the marker answers "where am I on this map",
    // which is a question about GEOGRAPHY, not about time. The case study's
    // whole point is to show what Cyclone Remal did over West Bengal, and the
    // viewer's own location is the most useful reference point for reading it
    // -- the Current Location panel already reports Remal's risk AT those
    // coordinates (Kolkata: 51% / 9.9 mm / 23%), so hiding the marker left the
    // panel describing a point the map refused to show.
    //
    // No live DATA is mixed in by doing this: the marker carries only the
    // coordinate, and its popup is labelled per-mode below.
    var loc = window.StormSenseUserLocation;
    if (!loc || loc.lat == null || loc.lon == null) return;

    // LIVE LOCATION COLOUR: violet (#a855f7).
    // Deliberately outside the risk ramp (emerald #10b981 / yellow #f59e0b /
    // orange #f97316 / red #ef4444) so the marker can never be read as a
    // severity level, and distinct from the cyan used everywhere for AI
    // forecast accents. It means "you are here" and nothing else.
    var icon = L.divIcon({
      className: "live-location-icon",
      html: '<div style="position:relative;width:26px;height:26px;display:flex;align-items:center;justify-content:center;">' +
              '<span style="position:absolute;width:24px;height:24px;border-radius:50%;background:#a855f7;opacity:0.30;"></span>' +
              '<span style="width:13px;height:13px;border-radius:50%;background:#a855f7;border:2.5px solid #ffffff;box-shadow:0 0 10px #a855f7;"></span>' +
            '</div>',
      iconSize: [26, 26],
      iconAnchor: [13, 13]
    });

    var accuracy = loc.accuracy_m != null
      ? '<div style="display:flex;justify-content:space-between;"><span>Accuracy:</span><strong>&plusmn;' + Math.round(loc.accuracy_m) + ' m</strong></div>'
      : '';

    // The marker shows a COORDINATE, which is mode-independent. The note below
    // states which field the surrounding map is painting, so the violet dot can
    // never be read as "live data inside the case study".
    var isHistMarker = (window.stormSenseMode === "historical");
    var modeNote = isHistMarker
      ? '<div style="margin-top:5px;padding-top:5px;border-top:1px solid #3b0764;color:#a78bfa;font-size:9px;line-height:1.4;">'
        + 'Your position, shown for geographic reference. The map around it is the '
        + 'Cyclone Remal case study (26 May 2024), not current conditions.</div>'
      : '';

    var popup =
      '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;border-radius:10px;min-width:190px;border:1px solid #7e22ce;">' +
        '<div style="font-weight:800;font-size:12px;margin-bottom:4px;color:#c084fc;">' +
          (isHistMarker ? 'YOUR LOCATION' : 'CURRENT LIVE LOCATION') + '</div>' +
        '<div style="display:flex;justify-content:space-between;"><span>Coordinates:</span><strong>' +
          Number(loc.lat).toFixed(4) + '°N, ' + Number(loc.lon).toFixed(4) + '°E</strong></div>' +
        accuracy +
        '<div style="display:flex;justify-content:space-between;"><span>Source:</span><strong>' + (loc.source || "device") + '</strong></div>' +
        modeNote +
      '</div>';

    var marker = L.marker([loc.lat, loc.lon], { icon: icon, zIndexOffset: 1000 }).bindPopup(popup);
    marker.addTo(mapInstance);
    mapInstance._liveLocationMarker = marker;
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

  // The public Grid On/Off control has been removed: it only ever applied in
  // historical mode, so it appeared broken in live mode. The underlying 0.25
  // degree model grid is unchanged and still drives every forecast value and
  // the risk field -- only the toggle button is gone.

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

  // -----------------------------------------------------------------------------
  // REMOVED: fabricated "Live Telemetry Station" marker.
  //
  // renderLiveStationMarker()/updateLiveStationPopup() used to drop a green
  // pulsing beacon at a HARDCODED [22.724, 88.479] and label it "LIVE STATION
  // TELEMETRY". Two things were wrong with it:
  //
  //   1. StormSense operates no surface station at that coordinate. The point
  //      was invented, so the marker claimed an observing asset that does not
  //      exist and pinned it to a location unrelated to the user.
  //   2. Its popup fell back to INVENTED values when the live feed was absent
  //      ("32.0 C", wind "16.0", pressure 1006, humidity 70) -- fabricated
  //      weather presented as station telemetry.
  //
  // Real surface observations are shown in the Current Location panel, tied to
  // the user's actual coordinates, with a real timestamp and age. The map's
  // only marker is the genuine live-location marker.
  // -----------------------------------------------------------------------------

  // -----------------------------------------------------------------------------
  // Unified Map Mode Switcher (Updates Layers on ONE Single Map Instance)
  // -----------------------------------------------------------------------------
  function updateMapMode(mode) {
    var map = window.stormSenseMap;
    if (!map) return;
    var isHistorical = mode === "historical";
    if (map.closePopup) map.closePopup();

    var hudTitle = document.getElementById("map-legend-title");
    var hudLead = document.getElementById("legend-lead-tag");

    if (isHistorical) {
      renderContinuousRiskSurface(map, window.currentLeadHours || 2);
      // The position marker is drawn in the case study too: it answers "where
      // am I on this map", which is geography, not time. It was previously
      // removed here, so the Current Location panel reported Remal's risk at
      // the viewer's coordinates while the map refused to show where that was.
      // renderLiveLocationMarker() labels the popup per-mode so no live data is
      // implied.
      renderLiveLocationMarker(map);
      if (hudTitle) hudTitle.textContent = "SEVERE WEATHER RISK (CASE STUDY)";
    } else {
      renderContinuousRiskSurface(map, window.currentLeadHours || 2);
      // The live location marker is the ONLY marker on this map, and it is
      // drawn only when the browser actually granted a real position.
      renderLiveLocationMarker(map);
      if (hudTitle) hudTitle.textContent = "SEVERE WEATHER RISK";
    }

    applyHorizonLegend();
    if (hudLead) {
      hudLead.textContent = horizonLabel();
      hudLead.className = "text-cyan-400 font-bold";
    }

    setTimeout(function () {
      map.invalidateSize();
    }, 100);
  }

  // NOW shows REAL observation timestamps and age. Nothing is simulated here:
  // if there is no live observation, it says so.
  function updateNowObservationBox() {
    var surface = window.StormSenseLiveSurface;
    var obsEl = document.getElementById("nv-obs-time");
    var ageEl = document.getElementById("nv-age");
    if (!obsEl || !ageEl) return;

    if (!surface || !surface.observed_at_utc) {
      obsEl.textContent = "unavailable";
      ageEl.textContent = "--";
      return;
    }
    var obs = new Date(surface.observed_at_utc);
    obsEl.textContent = formatIstClock(obs);
    // Prominent counterpart to the forecast valid-time stamp, so NOW and the
    // forecast horizons are read on the same visual footing.
    setText("nv-obs-time-big", formatIstStamp(obs));
    var ageSec = surface.data_age_seconds;
    if (ageSec == null) ageSec = Math.floor((Date.now() - obs.getTime()) / 1000);
    if (ageSec < 0) ageSec = 0;
    var m = Math.floor(ageSec / 60);
    var h = Math.floor(m / 60);
    ageEl.textContent = h > 0 ? (h + "h " + (m % 60) + "m ago") : (m <= 1 ? "just now" : m + "m ago");
    ageEl.className = ageSec > 3600 ? "text-amber-400" : "text-slate-300";
  }

  function horizonLabel() {
    var lead = window.currentLeadHours;
    return (lead === "now") ? "NOW" : "+" + (lead || 2) + "h";
  }

  // NOW and the future horizons carry different DATA SEMANTICS, and the legend
  // must say so: NOW is observation/monitoring, +2/+4/+6 are AI predictions.
  function applyHorizonLegend() {
    var isNow = window.currentLeadHours === "now";
    var title = document.getElementById("map-legend-title");
    var pills = document.getElementById("map-legend-pills");
    var sub = document.getElementById("map-legend-subtext");

    // The station/interpolation provenance line describes the LIVE OBSERVATION
    // layer (values measured at reporting stations, interpolated between them).
    // Live NOW now paints the model's lead=0 risk field and historical NOW
    // paints ERA5, so that line no longer describes either base surface and is
    // hidden: leaving it up would attribute the painted field to station
    // interpolation it did not come from. The observation stations themselves
    // remain a separate, separately-labelled layer.
    var prov = document.getElementById("obs-provenance");
    if (prov) prov.classList.add("hidden");

    // CHANGE 6: historical NOW now paints the STORMSENSE MODEL RISK field at
    // lead 0, exactly like live NOW, so its legend is the risk legend. The
    // observed ERA5 rainfall analysis is a separate product and keeps its own
    // blue/teal legend on the Radar & Satellite Feeds view.
    var isHistoricalNow = isNow && (window.stormSenseMode === "historical");

    if (isHistoricalNow) {
      if (title) title.textContent = "CYCLONE REMAL · MODEL T0 RISK (26 MAY 2024 12:00 UTC)";
      if (pills) {
        // Same probability bands as every other horizon
        // (NowcastService._level_for_prob: 0.25 / 0.50 / 0.75).
        pills.innerHTML =
          '<span class="px-2 py-0.5 bg-emerald-500 text-slate-950 rounded">&lt;25% Normal</span>' +
          '<span class="px-2 py-0.5 bg-amber-500 text-slate-950 rounded">25–50% Watch</span>' +
          '<span class="px-2 py-0.5 bg-orange-500 text-white rounded">50–75% Alert</span>' +
          '<span class="px-2 py-0.5 bg-red-500 text-white rounded">≥75% Warning</span>';
      }
      if (sub) {
        sub.textContent = "StormSense model risk valid at case-study T0 "
          + "· 26 May 2024 12:00 UTC · observed rainfall is a separate "
          + "product on the Radar view";
      }
    } else if (isNow) {
      // Live NOW paints the model's lead=0 risk field, so the legend describes
      // RISK on the same probability scale as +2/+4/+6 -- it must match what
      // renderContinuousRiskSurface() actually draws. The title states the
      // product and its validity explicitly: this is the model's own T0
      // analysis-valid risk, not an observation and not the +2h field.
      if (title) title.textContent = "NOW — MODEL T0 ANALYSIS-VALID RISK";
      if (pills) {
        // Same band edges as the forecast horizons
        // (NowcastService._level_for_prob: 0.25 / 0.50 / 0.75), because the same
        // surface renderer paints them.
        pills.innerHTML =
          '<span class="px-2 py-0.5 bg-emerald-500 text-slate-950 rounded">&lt;25% Normal</span>' +
          '<span class="px-2 py-0.5 bg-amber-500 text-slate-950 rounded">25–50% Watch</span>' +
          '<span class="px-2 py-0.5 bg-orange-500 text-white rounded">50–75% Alert</span>' +
          '<span class="px-2 py-0.5 bg-red-500 text-white rounded">≥75% Warning</span>' +
          '<span class="px-2 py-0.5 rounded text-white" style="background:#334155;">Obs station</span>' +
          '<span class="px-2 py-0.5 bg-purple-500 text-white rounded">Live location</span>';
      }
      if (sub) {
        sub.textContent = "StormSense model risk valid at T0 (lead 0) · separate "
          + "model run, not the +2h field · observation stations shown separately";
      }
    } else {
      if (title) title.textContent = "SEVERE WEATHER RISK";
      if (pills) {
        // The probability bands are the SAME edges the model classifier uses
        // (NowcastService._level_for_prob: 0.25 / 0.50 / 0.75), so the legend
        // and the painted field can never disagree. Wording matches the
        // backend stage names (NORMAL / WATCH / ALERT / WARNING).
        pills.innerHTML =
          '<span class="px-2 py-0.5 bg-emerald-500 text-slate-950 rounded">&lt;25% Normal</span>' +
          '<span class="px-2 py-0.5 bg-amber-500 text-slate-950 rounded">25–50% Watch</span>' +
          '<span class="px-2 py-0.5 bg-orange-500 text-white rounded">50–75% Alert</span>' +
          '<span class="px-2 py-0.5 bg-red-500 text-white rounded">≥75% Warning</span>';
      }
      if (sub) sub.textContent = "AI-derived severe weather risk · Not an official government warning";
    }
  }

  // -----------------------------------------------------------------------------
  // Interactive Map Click Handler (Continuous Surface Point Inspection)
  // -----------------------------------------------------------------------------
  // Distinguishes a genuine geographic rejection from a service failure, so the
  // popup never claims a valid West Bengal coordinate is out of bounds.
  function pointErrorHtml(title, detail) {
    return '<div style="font-family:monospace;font-size:11px;padding:8px;color:#f8fafc;background:#0f172a;' +
      'border-radius:8px;border:1px solid #334155;max-width:250px;">' +
      '<div style="color:#fbbf24;font-weight:bold;margin-bottom:4px;">' + title + '</div>' +
      '<div style="color:#94a3b8;font-size:10px;line-height:1.45;">' + detail + '</div>' +
      '</div>';
  }

  // ---------------------------------------------------------------------------
  // NOW observation-evidence block (backend: /api/nowcast/point -> now_evidence)
  //
  // Renders what radar and the nearest station are OBSERVING at this cell, each
  // with its own timestamp, beside the model's probability. It deliberately does
  // NOT alter the displayed probability: radar and station readings are not
  // inputs to SevereWeatherNetV2, and the measured agreement between the two
  // observation sources over West Bengal (r = 0.104) is far too weak to justify
  // any fusion weight. The block exists so a disagreement is VISIBLE rather than
  // silently averaged away -- see STORMSENSE_FULL_HANDOFF_REPORT.md section 8.
  //
  // Only present for live NOW (lead 0); the backend omits it on forecast
  // horizons (a current observation says nothing about a future hour) and
  // returns NOT_APPLICABLE in the frozen historical case study.
  // ---------------------------------------------------------------------------
  var NOW_VERDICT_STYLE = {
    AGREE:                 { color: "#34d399", label: "OBSERVATIONS AGREE" },
    OBSERVED_NOT_MODELLED: { color: "#f59e0b", label: "OBSERVED · NOT MODELLED" },
    MODELLED_NOT_OBSERVED: { color: "#38bdf8", label: "MODELLED · NOT YET OBSERVED" },
    NO_OBSERVATION:        { color: "#94a3b8", label: "NO OBSERVATION" }
  };

  function nowEvidenceHtml(ev) {
    if (!ev || !ev.verdict || ev.verdict === "NOT_APPLICABLE") return "";
    var st = NOW_VERDICT_STYLE[ev.verdict] || NOW_VERDICT_STYLE.NO_OBSERVATION;
    var o = ev.observed || {};
    var gaps = ev.product_time_gaps_hours || {};

    // Echo is a coverage FRACTION, not a rain rate and not a dBZ. Label it as
    // what it is. "no coverage" and "no echo" are different facts and must not
    // collapse into a single dash.
    var echoTxt;
    if (o.radar_has_coverage === false) {
      echoTxt = "no radar coverage";
    } else if (o.radar_echo_fraction == null) {
      echoTxt = "unavailable";
    } else {
      echoTxt = (o.radar_echo_fraction * 100).toFixed(0) + "% of cell";
    }
    var rainTxt = (o.station_rain_1h_mm == null)
      ? "unavailable"
      : o.station_rain_1h_mm + " mm/h";

    var lag = gaps.reference_minus_analysis;
    // When the lag cannot be computed, say so as a sentence rather than
    // splicing "unknown" into "is <lag> older than", which read as
    // "is unknown older than these observations".
    var lagLine = (lag == null)
      ? 'Model input age could not be determined'
      : 'Model input is <strong style="color:#cbd5e1;">' + lag.toFixed(1) + ' h</strong> older than these observations';

    return '' +
      '<div style="background:#0b1220;border:1px solid ' + st.color + '55;border-radius:6px;padding:6px;margin-bottom:6px;font-size:10px;">' +
        '<div style="color:' + st.color + ';font-weight:bold;margin-bottom:3px;display:flex;justify-content:space-between;align-items:center;">' +
          '<span>Observed now</span>' +
          '<span style="font-size:8px;letter-spacing:0.3px;">' + st.label + '</span>' +
        '</div>' +
        '<div style="display:flex;justify-content:space-between;"><span>Radar echo:</span><strong>' + echoTxt + '</strong></div>' +
        '<div style="display:flex;justify-content:space-between;color:#64748b;font-size:9px;margin-bottom:2px;"><span>radar scan</span><span>' + istClockFromIso(o.radar_time_utc) + '</span></div>' +
        '<div style="display:flex;justify-content:space-between;"><span>Station rain (1h):</span><strong>' + rainTxt + '</strong></div>' +
        '<div style="display:flex;justify-content:space-between;color:#64748b;font-size:9px;"><span>' + (o.nearest_station || "nearest station") + '</span><span>' + istClockFromIso(o.station_time_utc) + '</span></div>' +
        '<div style="border-top:1px solid #1e293b;margin-top:4px;padding-top:4px;color:#94a3b8;font-size:9px;line-height:1.4;">' +
          lagLine + ' (analysis ' + istClockFromIso((ev.model_input || {}).analysis_time_utc) + '). ' +
          'Observations are shown for context and do not change the model probability.' +
        '</div>' +
      '</div>';
  }

  // ISO-8601 (UTC) -> "HH:MM IST", timezone-independent. Returns an em dash for
  // a missing time rather than inventing "now".
  function istClockFromIso(iso) {
    if (!iso) return "—";
    var d = new Date(iso);
    if (isNaN(d.getTime())) return "—";
    var ist = new Date(d.getTime() + (5.5 * 3600000));
    return String(ist.getUTCHours()).padStart(2, "0") + ":" +
           String(ist.getUTCMinutes()).padStart(2, "0") + " IST";
  }

  // What the LIVE "NOW" field actually is. The model has no lead-0 head, so NOW
  // is the +2h output of an inference run on a 2-hour-earlier input window. With
  // ERA5 (Historical) that lands exactly on t0. With GFS (Live) analyses arrive
  // only every 6h, so the earlier window usually resolves to the SAME cycle and
  // NOW comes out identical to +2h, valid at analysis+2h -- typically hours
  // behind wall clock. Shown so NOW is never read as a real-time observation.
  function nowProvenanceHtml(pv) {
    if (!pv || pv.is_true_t0_analysis !== false) return "";
    var vt = istClockFromIso(pv.field_valid_time_utc);
    var age = (pv.field_age_vs_reference_hours == null)
      ? null : pv.field_age_vs_reference_hours.toFixed(1) + " h";
    var same = pv.identical_to_plus_2h === true;
    var accent = same ? "#f59e0b" : "#94a3b8";
    return '' +
      '<div style="background:#0b1220;border:1px solid ' + accent + '44;border-radius:6px;padding:6px;margin-bottom:6px;font-size:9px;line-height:1.45;color:#94a3b8;">' +
        '<div style="color:' + accent + ';font-weight:bold;font-size:10px;margin-bottom:2px;">NOW field provenance</div>' +
        '<div style="display:flex;justify-content:space-between;"><span>Field valid at</span><strong style="color:#cbd5e1;">' + vt + '</strong></div>' +
        (age ? '<div style="display:flex;justify-content:space-between;"><span>Behind wall clock by</span><strong style="color:#cbd5e1;">' + age + '</strong></div>' : '') +
        (same ? '<div style="margin-top:3px;">Identical to the +2 h field: GFS publishes analyses every 6 h, so the 2-hour-earlier input window resolved to the same cycle. Read NOW as the earliest available forecast step, not a real-time observation.</div>' : '') +
      '</div>';
  }

  function handleMapClick(e) {
    if (!e || !e.latlng) return;

    var lat = e.latlng.lat;
    var lon = e.latlng.lng;
    var lead = apiLeadHours();
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
        // A failed/empty response is a REQUEST failure, not a statement about
        // geography. Reporting it as "outside model domain" previously made a
        // transient backend error look like a coordinate rejection and sent
        // debugging in entirely the wrong direction.
        if (!data) {
          popup.setContent(pointErrorHtml(
            "Could not reach the forecast service",
            "The point query returned no data. The location was not evaluated."
          ));
          return;
        }
        // The ONLY genuine outside-region case: the backend evaluated the
        // coordinate against the official state boundary and rejected it.
        if (data.inside_monitored_region === false) {
          popup.setContent(pointErrorHtml(
            "Outside the monitored region",
            Number(lat).toFixed(3) + "°N, " + Number(lon).toFixed(3) +
            "°E lies outside West Bengal, so no forecast is issued for it."
          ));
          return;
        }
        if (data.status === "unavailable") {
          popup.setContent(pointErrorHtml(
            "Forecast not available yet",
            data.message || "The live forecast state has not been ingested yet."
          ));
          return;
        }
        if (!data.predictions && data.status !== "success") {
          popup.setContent(pointErrorHtml(
            "No forecast values returned",
            "The service responded but carried no prediction for this point."
          ));
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
          ? '<span style="font-size:9px;color:#34d399;background:#064e3b40;border:1px solid #05966960;padding:2px 6px;border-radius:4px;">LIVE FORECAST</span>'
          : '<span style="font-size:9px;color:#38bdf8;background:#0369a140;border:1px solid #38bdf860;padding:2px 6px;border-radius:4px;">HISTORICAL CASE STUDY</span>';

        var liveSection = "";
        if (isLive && liveTelemetry) {
          liveSection =
            '<div style="background:#064e3b40;border:1px solid #05966960;border-radius:6px;padding:6px;margin-bottom:6px;font-size:10px;">' +
              '<div style="color:#34d399;font-weight:bold;margin-bottom:2px;display:flex;justify-content:space-between;">' +
                '<span>Live Station Telemetry (t=0):</span><span style="font-size:9px;color:#6ee7b7;">Live Station</span>' +
              '</div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Temp / Humidity:</span><strong>' + (liveTelemetry.temperature_c != null ? liveTelemetry.temperature_c + '°C' : '—') + ' / ' + (liveTelemetry.humidity_pct != null ? liveTelemetry.humidity_pct + '%' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Rain Gauge (1h):</span><strong>' + (liveTelemetry.rainfall_mm != null ? liveTelemetry.rainfall_mm + ' mm' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Surface Wind:</span><strong>' + (liveTelemetry.wind_kmh != null ? liveTelemetry.wind_kmh + ' km/h' : '—') + '</strong></div>' +
            '</div>';
        } else if (!isLive && t0) {
          liveSection =
            '<div style="background:#1e293b;border-radius:6px;padding:6px;margin-bottom:6px;font-size:10px;">' +
              // These are the model's INITIAL CONDITIONS at t=0. They are
              // identical across +2/+4/+6 by design -- the same analysis
              // initialises every horizon -- so they must not be read as
              // future observations at the selected valid time.
              '<div style="color:#38bdf8;font-weight:bold;margin-bottom:2px;">Model Initial Conditions (t=0):</div>' +
              '<div style="color:#94a3b8;font-size:9px;margin-bottom:3px;">Analysis state that initialises every horizon — not a forecast for the valid time</div>' +
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
            nowEvidenceHtml(data.now_evidence) +
            nowProvenanceHtml(data.now_provenance) +
            liveSection +
            '<div style="background:#0f172a;border-radius:6px;padding:6px;border:1px solid #1e293b;margin-bottom:6px;font-size:11px;">' +
              '<div style="color:#e2e8f0;font-weight:bold;margin-bottom:4px;font-size:10px;">StormSense Predictions:</div>' +
              '<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Thunderstorm Risk:</span><strong style="color:' + color + ';">' + (p.thunderstorm_prob_pct != null ? p.thunderstorm_prob_pct + '%' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>Heavy Rain (3h):</span><strong>' + (rainVal != null ? rainVal + ' mm' : '—') + '</strong></div>' +
              '<div style="display:flex;justify-content:space-between;"><span>Flash Flood Proxy:</span><strong style="color:#fbbf24;">' + (p.flash_flood_proxy_pct != null ? p.flash_flood_proxy_pct + '%' : '—') + '</strong></div>' +
            '</div>' +
            '<div style="font-size:9px;color:#64748b;border-top:1px solid #1e293b;padding-top:4px;margin-top:4px;display:flex;justify-content:space-between;">' +
              '<span>' + modelName + '</span>' +
              '<span>Valid: ' + validTime + '</span>' +
            '</div>' +
          '</div>';

        popup.setContent(content);
      })
      .catch(function (err) {
        console.warn("Point query error:", err);
        popup.setContent(pointErrorHtml(
          "Point query failed",
          "The forecast service could not be reached. This is a connection "
          + "problem, not a statement about this location."
        ));
      });
  }

  // -----------------------------------------------------------------------------
  // Main Map Initializer (Google Maps 2D base · StormSense overlays on top)
  // -----------------------------------------------------------------------------
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

    // BASE MAP: Google Maps 2D (constructed by L.map above).
    //
    // An Esri "World Dark Gray Base" tile layer used to be stacked here. It was
    // the base map before the Google migration, and it kept loading afterwards
    // as an opaque overlay ON TOP of Google -- so Google initialised, paid for
    // the tiles, and was then almost entirely hidden. Removing it restores the
    // detail the Google base is here to provide (place names, district labels,
    // bilingual rendering). The dark cartographic look is preserved by the
    // Google style applied in the shim, so nothing visual is lost.

    // REMOVED: a decorative red "IMD Kolkata DWR" marker used to be pinned here
    // at a hardcoded [22.65, 88.45]. StormSense does not ingest any DWR feed,
    // so the marker asserted a data source that does not exist and put a red
    // dot -- a risk-scale colour -- on the risk map. The only marker this map
    // is allowed to carry is the user's real live location.

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

    // INTERACTION BUDGET.
    //
    // These two constraints previously made the map almost unusable:
    //
    //   setMaxBounds(wbBounds.pad(0.35)) produced a restriction box barely
    //   wider than the viewport at the framing zoom, so Google clamped every
    //   horizontal drag back to the same longitude -- east/west panning was
    //   dead (measured: dLng exactly 0.0 over a 120px drag).
    //
    //   setMinZoom(getBoundsZoom(wbBounds)) set minZoom to the SAME zoom the
    //   map is framed at. Google refuses a zoom change that would leave the
    //   restriction, and with minZoom == current zoom there was no room to go
    //   out; combined with the tight box, scroll-wheel zoom did nothing at all
    //   in either direction.
    //
    // The map still opens framed on West Bengal (fitBounds above, and the WB
    // Focus control re-frames it on demand). The constraints are widened so
    // navigation actually works: a generous pan box that still stops the user
    // drifting to another continent, and a minZoom one step below the framing
    // zoom so zoom-out has somewhere to go.
    var framingZoom = map.getBoundsZoom(wbBounds);
    map.setMaxBounds(wbBounds.pad(1.4));
    map.setMinZoom(Math.max(4, framingZoom - 2));


    // NO RainViewer radar layer here. The Interactive Nowcasting Map renders
    // StormSense's own state only: observed conditions at NOW, and the model's
    // continuous risk field at +2/+4/+6h. Overlaying a third-party
    // precipitation mosaic on top of it made external radar imagery look like
    // StormSense output and conflated precipitation with severe-weather risk.
    // RainViewer remains available on the dedicated Radar & Satellite Feeds
    // view (initRadarMap), which is where a radar mosaic belongs.
    return map;
  }



  var radarLayer = null;

  // RainViewer is permitted on the Radar & Satellite Feeds map ONLY.
  // This guard is deliberate: it makes it impossible to reintroduce the radar
  // mosaic onto the Interactive Nowcasting Map by accident from any call site.
  function isRadarFeedsMap(map) {
    return !!map && !!map.getContainer &&
           map.getContainer().id === "radar-map-container";
  }

  async function loadRainViewerRadar(map) {
    if (window.stormSenseMode === "historical") {
      console.warn("loadRainViewerRadar: blocked in historical mode to prevent data contamination.");
      return;
    }
    if (!isRadarFeedsMap(map)) {
      console.warn(
        "loadRainViewerRadar: refused -- RainViewer is only allowed on the " +
        "Radar & Satellite Feeds map, never on the Interactive Nowcasting Map."
      );
      return;
    }

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

        var sptLive = document.getElementById("spatial-panel-title");
        if (sptLive) sptLive.textContent = "AI Spatial Forecast";
        if (statusEl) statusEl.textContent = "RADAR READY";
        // Name the product precisely. This is an OBSERVED composite radar
        // mosaic, not StormSense's AI forecast and not a rainfall accumulation.
        if (infoEl) infoEl.textContent = "Observed radar mosaic · RainViewer composite · ~10 min frames";
        if (updateEl) {
            // Radar scan time in IST, matching every other timestamp in the app.
            // toLocaleTimeString() rendered this in the VIEWER's timezone, so on
            // a non-IST machine the radar clock silently disagreed with the rest
            // of the page. The observation latency is disclosed rather than
            // hidden: this is an observed scan, and it is already minutes old.
            var scanTime = new Date(latestFrame.time * 1000);
            var ageMin = Math.max(0, Math.round((Date.now() - scanTime.getTime()) / 60000));
            updateEl.textContent = "Last Scan: " + formatIstClock(scanTime)
                + " · observed " + (ageMin <= 1 ? "just now" : ageMin + " min ago");
        }
    } catch (error) {
        console.error("RainViewer radar error:", error);
        if (statusEl) statusEl.textContent = "RADAR OFFLINE";
        if (infoEl) infoEl.textContent = "Radar feed unavailable";
        if (updateEl) updateEl.textContent = "Connection failed";
    }
  }

  // ---------------------------------------------------------------------------
  /**
   * Re-apply the radar panel's mode semantics to an ALREADY-CREATED radar map.
   *
   * initRadarMap() returns early when the map exists, so its historical branch
   * never ran on a later mode switch: a map built in live mode kept showing
   * "RADAR READY" and the RainViewer label after switching to the frozen case
   * study, and kept a live radar tile layer attached over 2024 data.
   */
  function applyRadarModeSemantics() {
    var map = window.stormSenseRadarMap;
    var isHist = (window.stormSenseMode === "historical");
    var spt = document.getElementById("spatial-panel-title");
    var rs = document.getElementById("radar-status");
    var ri = document.getElementById("radar-info");
    var ru = document.getElementById("radar-last-update");

    if (isHist) {
      // Detach any live radar tiles: today's radar must never sit over Remal.
      if (map) {
        map.eachLayer(function (l) {
          if (l && l._url && String(l._url).indexOf("rainviewer") >= 0) map.removeLayer(l);
        });
      }
      if (spt) spt.textContent = "Cyclone Remal · Observed Rainfall Analysis (t=0)";
      if (rs) {
        rs.textContent = "ERA5 OBSERVED";
        rs.className = "text-[10px] font-mono bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/40 font-bold";
      }
      if (ri) ri.textContent = "ERA5 reanalysis · observed rainfall rate (mm/h) · not a forecast";
      if (ru) ru.textContent = "Analysis 26 May 2024 · 12:00 UTC";
    } else {
      if (spt) spt.textContent = "AI Spatial Forecast & Live Radar";
      if (map && typeof loadRainViewerRadar === "function") loadRainViewerRadar(map);
    }
  }
  window.applyRadarModeSemantics = applyRadarModeSemantics;

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

    // BASE MAP: Google Maps 2D. The Esri dark-gray tile layer that used to be
    // stacked here is removed for the same reason as on the nowcasting map --
    // it covered the Google base it was drawn over. RainViewer is unaffected:
    // it is a SEPARATE radar overlay, added by loadRainViewerRadar() below, and
    // it remains the only tile overlay on this map.

    window.stormSenseRadarMap = map;

    if (window.StormSenseStateBoundaryData) {
      renderStateBoundary(map, window.StormSenseStateBoundaryData);
    }
    if (window.StormSenseBoundariesData) {
      renderDistrictBoundaries(map, window.StormSenseBoundariesData);
    }
    if (window.stormSenseMode === "historical") {
      // This panel must NOT duplicate the Dashboard. The Dashboard map shows the
      // model's severe-weather PROBABILITY forecast for the selected horizon;
      // here we draw the OBSERVED ERA5 rainfall field at t=0 -- a different
      // quantity (mm/hour, not %), a different time (analysis, not +Nh), and a
      // different palette (blue/teal observation ramp, not green->red risk).
      renderHistoricalAnalysisSurface(map);
    }

    if (map._stateBoundaryLayer) {
      map.fitBounds(map._stateBoundaryLayer.getBounds(), { padding: [20, 20] });
    } else {
      map.fitBounds([[21.5394, 86.6103], [26.9960, 89.8828]], { padding: [15, 15] });
    }


    if (window.stormSenseMode === "historical") {
        // HONEST LABELLING. The field painted above is the StormSense MODEL
        // FORECAST for the case study, not archived radar -- StormSense holds
        // no archived Remal radar imagery. This box previously read "CYCLONE
        // REMAL RADAR ARCHIVE", presenting model output as a radar observation.
        // No live radar is fetched in historical mode (the else-branch below).
        var msg = L.divIcon({
            className: 'radar-archive-msg',
            html: '<div style="color:#00e5ff; background:rgba(0,0,0,0.85); padding:10px; border:1px solid #00e5ff; text-align:center; font-family:monospace; line-height:1.45;">'
                + '<b>OBSERVED RAINFALL FIELD · t=0</b><br/>'
                + 'Cyclone Remal · analysis 26 May 2024 12:00 UTC<br/>'
                + '<span style="opacity:.85">ERA5 reanalysis · mm/hour · not a forecast</span></div>',
            iconSize: [340, 74]
        });
        // Kept north of ~26N: Remal's rainfall sits over the southern coast, and
        // a caption centred mid-domain covered the very field it describes.
        L.marker([26.6, 87.2], {icon: msg}).addTo(map);
        // Radar HUD must not imply a live feed is loading in a frozen replay.
        // Panel identity must describe what is actually rendered: the
        // StormSense historical model field, not a radar product.
        var spt = document.getElementById("spatial-panel-title");
        if (spt) spt.textContent = "Cyclone Remal · Observed Rainfall Analysis (t=0)";
        var rs = document.getElementById("radar-status");
        if (rs) {
          rs.textContent = "ERA5 OBSERVED";
          rs.className = "text-[10px] font-mono bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/40 font-bold";
        }
        var ri = document.getElementById("radar-info");
        if (ri) ri.textContent = "ERA5 reanalysis · observed rainfall rate (mm/h) · not a forecast";
        var ru = document.getElementById("radar-last-update");
        if (ru) ru.textContent = "Analysis 26 May 2024 · 12:00 UTC";
    } else {
        if (!radarLayer) {
            loadRainViewerRadar(map);
        }
    }
    return map;
  }


  // -----------------------------------------------------------------------------
  // View Switcher & Horizon Controller
  // -----------------------------------------------------------------------------
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
      // Built on first visit, not at page load, so RainViewer tiles are only
      // ever fetched when the user is actually looking at the radar view.
      if (!window.stormSenseRadarMap) {
        // Gated on the SDK for the same reason as the dashboard map: it now
        // loads asynchronously and may be on a failover key.
        whenMapsReady(function () {
          initRadarMap(window.StormSenseDashboardData || null);
        });
      } else {
        // Map already exists from an earlier visit: re-apply the CURRENT mode's
        // semantics, since initRadarMap() short-circuits and would leave live
        // radar labelling (and tiles) in place over a historical case study.
        applyRadarModeSemantics();
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
    if (label === 'now' || leadHours === 0 || label === 'NOW') {
      // NOW is OBSERVATION / MONITORING, not an AI forecast. The predicted risk
      // field is therefore hidden and the observational layers are shown, with
      // the legend and validity box saying which is which.
      window.currentLeadHours = 'now';
      var nowBox = document.getElementById('now-validity-box');
      var fcBox = document.getElementById('forecast-validity-box');
      if (nowBox) { nowBox.classList.remove('hidden'); nowBox.classList.add('flex'); }
      if (fcBox) { fcBox.classList.add('hidden'); fcBox.classList.remove('flex'); }

      // REMOVE the forecast surface rather than making it transparent. An
      // opacity-0 overlay is still attached to the map, so any later repaint
      // could bring the +2h prediction back while NOW is selected. Removing it
      // makes "NOW shows no forecast field" a structural fact.
      if (window.stormSenseMap) {
        // Route through the SINGLE horizon renderer rather than picking a layer
        // here. This branch previously called renderObservationSurface()
        // directly, which returns early in historical mode -- so selecting NOW
        // in the case study removed the forecast layer and drew nothing,
        // leaving the map blank. renderContinuousRiskSurface() owns the
        // mode-aware choice: live NOW -> observations, historical NOW -> the
        // event's t=0 analysis field.
        renderContinuousRiskSurface(window.stormSenseMap, 'now');
        renderLiveLocationMarker(window.stormSenseMap);
        window.stormSenseMap.closePopup();
      }
      applyHorizonLegend();
      var nowTag = document.getElementById("legend-lead-tag");
      if (nowTag) nowTag.textContent = "NOW";

      document.querySelectorAll(".horizon-btn").forEach(function (b) {
        b.classList.remove("bg-cyan-600", "bg-red-600", "text-white", "font-bold");
        b.classList.add("text-slate-400", "font-medium");
        if (b.textContent.trim() === "NOW") {
          b.classList.add("bg-cyan-600", "text-white", "font-bold");
          b.classList.remove("text-slate-400", "font-medium");
        }
      });
      // Re-stamp the temporal banner at LEAD ZERO. Without this the NOW branch
      // returned early and the "Valid ..." line kept the previously selected
      // horizon's value -- historical NOW displayed "Valid 18:00 UTC" (a stale
      // +6h) above its own 12:00 analysis state.
      var nowSummary = window.StormSenseNowcastData;
      var nowIssue = (nowSummary && nowSummary.issue_time)
        || (nowSummary && nowSummary.valid_time)
        || null;
      if (nowIssue) {
        updateDynamicTimes(nowIssue, "now", nowSummary && nowSummary.analysis_time_iso);
      }
      // Repaint the LOWER physical-condition cards for the NOW/T0 state.
      // Without this they kept whatever the previously selected horizon wrote
      // -- selecting NOW in the case study left "+6h" cards carrying a live
      // 2026 valid time above 2024 data.
      if (window.stormSenseMode === "historical") {
        paintHistoricalConditions(window.StormSenseNowcastData);
      } else if (window.StormSenseLiveSurface) {
        paintDashboardLive(window.StormSenseLiveSurface.observations || {},
                           window.StormSenseLiveSurface);
      }
      updateNowObservationBox();

      // HORIZON SYNCHRONISATION AT NOW.
      //
      // This branch used to return here without fetching anything, so the four
      // hazard KPI cards, the district advisories, the high-risk cells and XAI
      // all kept whatever the PREVIOUSLY selected horizon had written. Measured
      // symptom: selecting NOW after +2h left the cards reading the +2h values
      // (31% / 12.0 mm / 19%) while the map had correctly switched to lead=0,
      // whose real values are different (66% / 8.3 mm / 13%).
      //
      // NOW is lead 0 at the API, so it can and must refresh through the same
      // path as every other horizon. Only the map surface is special-cased
      // (handled above by renderContinuousRiskSurface(map,'now')), which is why
      // this block deliberately does not repaint it.
      var nowModeQS = "&mode=" + (window.stormSenseMode || "live");
      return Promise.all([
        fetch(API_BASE + "/api/nowcast/summary?lead=0" + nowModeQS + districtQS()).then(parseJson).catch(function () { return null; }),
        fetch(API_BASE + "/api/nowcast/districts?lead=0" + nowModeQS).then(parseJson).catch(function () { return null; }),
        fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=0&top_k=8" + nowModeQS).then(parseJson).catch(function () { return null; }),
        fetch(API_BASE + "/api/nowcast/xai?lead=0" + nowModeQS).then(parseJson).catch(function () { return null; })
      ]).then(function (res) {
        var summary = res[0], districts = res[1], cells = res[2], xai = res[3];
        if (summary && summary.hazards) {
          window.StormSenseCurrentSummary = summary;
          window.StormSenseNowcastData = summary;
          paintForecastCards(summary);
          if (summary.timeline) renderNowcast(summary.timeline);
          applyRegionState(summary);
          applyFreshnessState(summary);
          if (summary.issue_time) {
            updateDynamicTimes(summary.issue_time, "now", summary.analysis_time_iso);
          }
        }
        if (districts && districts.length) {
          window.StormSenseDistrictsData = districts;
          renderDistricts(adaptDistricts(districts, 0,
            (summary && summary.issue_time) || null));
          window.renderBulletinsFromDistricts(districts, summary);
        }
        if (cells) {
          window.StormSenseHighRiskCells = cells;
          renderHighRiskCells(cells);
        }
        if (xai && typeof renderXai === "function") renderXai(xai);
        // Re-apply the NOW legend last: paintForecastCards/updateDynamicTimes
        // above can restamp the lead tag from the summary payload.
        applyHorizonLegend();
        var tagEl = document.getElementById("legend-lead-tag");
        if (tagEl) tagEl.textContent = "NOW";
      }).catch(function () { return null; });
    }

    var lead = leadHours;
    if (!lead) {
      if (label === "+2h") lead = 2;
      else if (label === "+4h") lead = 4;
      else if (label === "+6h") lead = 6;
      else lead = 2;
    }
    if (lead !== 2 && lead !== 4 && lead !== 6) {
      lead = lead < 3 ? 2 : (lead < 5 ? 4 : 6);
    }

    window.currentLeadHours = lead;
    // Retitle the condition panel NOW, not after the six-request Promise.all
    // below resolves. Until this ran, the panel kept the "Current Observations"
    // heading and NOW's observed rainfall for the whole fetch window, so a
    // freshly clicked +2h briefly presented live observations as the forecast.
    paintHorizonConditions();
    document.getElementById('now-validity-box').classList.add('hidden');
    document.getElementById('forecast-validity-box').classList.remove('hidden');
    
    if (window.stormSenseMap && window.stormSenseMap._riskSurfaceOverlay) {
      window.stormSenseMap._riskSurfaceOverlay.setOpacity(0.85);
    }
    
    if (window.stormSenseMap && window.stormSenseMap.closePopup) window.stormSenseMap.closePopup();
    applyHorizonLegend();

    document.querySelectorAll(".horizon-btn").forEach(function (b) {
      b.classList.remove("bg-cyan-600", "bg-red-600", "text-white", "font-bold");
      b.classList.add("text-slate-400", "font-medium");
      var txt = b.textContent.trim();
      if (txt === "+" + lead + "h") {
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

    return Promise.all([
      fetch(API_BASE + "/api/nowcast/summary?lead=" + lead + modeQS + districtQS()).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/districts?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/risk-map?lead=" + lead + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/high-risk-cells?lead=" + lead + "&top_k=8" + modeQS).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/xai" + "?mode=" + (window.stormSenseMode || "live") + "&lead=" + apiLeadHours()).then(parseJson).catch(function () { return null; }),
      fetch(API_BASE + "/api/nowcast/thermodynamics" + "?mode=" + (window.stormSenseMode || "live")).then(parseJson).catch(function () { return null; })
    ])
      .then(function (results) {
        var summary = results[0];
        window.StormSenseCurrentSummary = summary;
        var districts = results[1];
        var riskMap = results[2];
        var cells = results[3];

        var issueTime = (summary && summary.issue_time) || "2024-05-26T12:00:00Z";
        updateDynamicTimes(issueTime, lead, summary && summary.analysis_time_iso);
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
          window.renderBulletinsFromDistricts(districts, summary);
        } else {
          window.renderBulletinsFromDistricts([], summary);
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

        // §5 TEMPORAL CONSISTENCY: the Current Location panel must follow the
        // selected horizon too. It previously kept whichever horizon was active
        // when geolocation first resolved, so selecting +4h left the local
        // meters showing the +2h forecast while the rest of the page moved on.
        // applyLiveLocation() re-queries /api/nowcast/point with the CURRENT
        // lead and repaints the meters from those coordinates.
        if (window.StormSenseUserLocation && typeof window.applyLiveLocation === "function") {
          window.applyLiveLocation(window.StormSenseUserLocation)
            .then(function () { paintHorizonConditions(); });
        } else {
          paintHorizonConditions();
        }

        window.showToast("StormSense Nowcast updated: +" + lead + "h lead time", "check_circle");
      })
      .catch(function (err) {
        console.error("Failed to load horizon nowcast:", err);
        window.showToast("Failed to fetch +" + lead + "h nowcast", "warning");
      });
  };

  // -----------------------------------------------------------------------------
  // Interactive Modals, Toasts & User Utilities
  // -----------------------------------------------------------------------------
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
    "West Bengal Coastal Delta": [22.724, 88.479],
    "Kolkata Metropolitan Area": [22.572, 88.363]
  };

  // ---------------------------------------------------------------------------
  // Live location acquisition and the Current Location panel
  // ---------------------------------------------------------------------------
  // Resolves the CURRENT location from the browser. Nothing is invented: if the
  // device will not provide a position, the panel reports that honestly instead
  // of presenting the target-region centroid as "your location".
  function acquireLiveLocation() {
    return new Promise(function (resolve) {
      if (!("geolocation" in navigator)) {
        window.StormSenseUserLocation = null;
        resolve(null);
        return;
      }
      navigator.geolocation.getCurrentPosition(
        function (pos) {
          window.StormSenseUserLocation = {
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
            accuracy_m: pos.coords.accuracy,
            source: "device GPS",
            acquired_at: new Date().toISOString()
          };
          resolve(window.StormSenseUserLocation);
        },
        function (err) {
          console.info("Live location unavailable:", err && err.message);
          window.StormSenseUserLocation = null;
          resolve(null);
        },
        { timeout: 8000, enableHighAccuracy: true, maximumAge: 60000 }
      );
    });
  }

  // Applies the live location to BOTH the map marker and the Current Location
  // panel, so the two always describe the same coordinates.
  window.applyLiveLocation = function (loc) {
    var nameEl = document.getElementById("profile-district");
    var coordEl = document.getElementById("profile-coords");
    var metaEl = document.getElementById("profile-meta");

    if (!loc || loc.lat == null) {
      if (nameEl) nameEl.textContent = "Location unavailable";
      if (coordEl) coordEl.textContent = "";
      if (metaEl) {
        metaEl.textContent = "Current location could not be determined. "
          + "Showing state-wide monitoring instead.";
      }
      if (window.stormSenseMap) removeLiveLocationMarker(window.stormSenseMap);
      // The COORDINATES are unknown, but the conditions cards are not
      // location-exclusive -- they fall back to the state-wide query. Returning
      // here without painting them left Temperature / Rainfall / Humidity /
      // Wind frozen on their "..." placeholder for anyone who denies or cannot
      // provide geolocation, which is the default in most privacy settings.
      // Paint them from the state-wide fallback so the panel always states
      // either a real value or an explicit "—", never a loading ellipsis.
      paintHorizonConditions();
      // The three risk meters ARE location-exclusive -- they describe the
      // viewer's own coordinates -- so there is no honest state-wide value to
      // substitute. Reset them to "—" instead of leaving the "..." placeholder,
      // which read as "still loading" forever for anyone who denies location.
      applyPointRiskMeters(null);
      return Promise.resolve(null);
    }

    if (coordEl) {
      coordEl.textContent = Number(loc.lat).toFixed(4) + "°N, " + Number(loc.lon).toFixed(4) + "°E";
    }
    if (window.stormSenseMap) renderLiveLocationMarker(window.stormSenseMap);

    // Ask the backend which district (if any) the live coordinates fall in, and
    // populate the panel from the forecast AT THOSE COORDINATES.
    var lead = apiLeadHours();
    return fetch(API_BASE + "/api/nowcast/point?lat=" + loc.lat.toFixed(4)
        + "&lon=" + loc.lon.toFixed(4) + "&lead=" + lead
        + "&mode=" + (window.stormSenseMode || "live"))
      .then(parseJson)
      .then(function (d) {
        if (!d) return null;
        if (d.inside_monitored_region === false) {
          if (nameEl) nameEl.textContent = "Outside West Bengal";
          if (metaEl) {
            metaEl.textContent = "Your current location is outside the monitored "
              + "region, so no local forecast is issued for it.";
          }
          return d;
        }
        if (nameEl) nameEl.textContent = d.district || "Current location";
        if (metaEl) metaEl.textContent = "Observed conditions and AI forecast risk at your coordinates";
        window.StormSenseUserPoint = d;
        // Bind the risk meters to the forecast AT THESE COORDINATES. They
        // previously kept the STATE-WIDE values painted by paintForecastCards(),
        // so a user in South 24 Parganas saw the statewide peak district's
        // numbers (e.g. Cooch Behar's) presented as their own local risk.
        applyPointRiskMeters(d);
        return d;
      })
      .catch(function (e) {
        console.warn("Could not resolve live location forecast:", e);
        return null;
      });
  };

  window.handleMyLocationClick = function () {
    var ping = document.getElementById("my-location-ping");
    if (ping) ping.classList.remove("hidden");
    window.showToast("Locating your GPS position...", "radar");

    // Single shared location path -- the same state the Current Location panel
    // and the map marker both read, so they can never disagree.
    acquireLiveLocation().then(function (loc) {
      if (ping) ping.classList.add("hidden");
      if (!loc) {
        window.showToast("Current location unavailable. The monitored region is unchanged.", "warning");
        window.applyLiveLocation(null);
        return;
      }
      if (window.stormSenseMap) {
        window.stormSenseMap.flyTo([loc.lat, loc.lon], 11);
      }
      window.applyLiveLocation(loc);
      window.showToast(
        "Current location: " + loc.lat.toFixed(3) + "°N, " + loc.lon.toFixed(3) + "°E",
        "check_circle"
      );
    });
  };

  window.stubDemoAction = function (label) {
    window.showToast("Operational module: " + label + " executed.", "check_circle");
  };

  // -----------------------------------------------------------------------------
  // DOM Initialization
  // -----------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // Google Maps readiness gate.
  //
  // index.html loads the SDK asynchronously and fails over between keys when one
  // hits its daily quota, so `google.maps` can appear late (or, if every key is
  // exhausted, never). Callers register here instead of assuming the SDK exists.
  // ---------------------------------------------------------------------------
  function whenMapsReady(fn) {
    if (window.STORMSENSE_MAPS_READY && typeof google !== "undefined" && google.maps) {
      fn();
      return;
    }
    if (window.STORMSENSE_MAPS_EXHAUSTED) {
      showMapsUnavailable();
      return;
    }
    document.addEventListener("stormsense:maps-ready", function once() {
      document.removeEventListener("stormsense:maps-ready", once);
      fn();
    });
    document.addEventListener("stormsense:maps-exhausted", function once() {
      document.removeEventListener("stormsense:maps-exhausted", once);
      showMapsUnavailable();
    });
  }

  // Replace Google's bare "Oops! Something went wrong." panel with an accurate
  // statement of what failed. The rest of the dashboard keeps working: the base
  // map is a rendering surface, not a data source, so risk values, cards,
  // advisories and popup data are unaffected by a Maps quota exhaustion.
  function showMapsUnavailable() {
    var st = window.STORMSENSE_MAPS_KEY_STATE || { total: 0 };
    ["map-radar-placeholder", "radar-map-container"].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.innerHTML =
        '<div style="height:100%;display:flex;align-items:center;justify-content:center;padding:18px;' +
        'background:#0f172a;border-radius:10px;">' +
          '<div style="font-family:monospace;font-size:12px;color:#f8fafc;max-width:430px;text-align:center;">' +
            '<div style="color:#fbbf24;font-weight:bold;margin-bottom:6px;">BASE MAP UNAVAILABLE</div>' +
            '<div style="color:#94a3b8;font-size:11px;line-height:1.55;">' +
              'All ' + st.total + ' configured Google Maps key(s) hit their daily quota or were rejected, ' +
              'so the base map cannot be drawn. ' +
              '<strong style="color:#cbd5e1;">This is a Google Maps account limit, not a StormSense data problem.</strong><br><br>' +
              'Forecast values, hazard cards, district advisories and point queries are unaffected ' +
              'and remain live — only the map imagery is missing.' +
            '</div>' +
          '</div>' +
        '</div>';
    });
  }

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

      // Initialize the dashboard's Interactive Nowcasting Map.
      //
      // The Maps SDK is now loaded ASYNCHRONOUSLY with per-key failover (see the
      // loader in index.html), because Google's daily quota is per key and an
      // exhausted key fails inside the SDK rather than as an HTTP error. That
      // means `google.maps` may not exist yet when DOMContentLoaded fires, and
      // may arrive later on a second key. Build the map only once the SDK is
      // genuinely ready, so a failover still produces a working map instead of
      // a silently dead one.
      whenMapsReady(function () { initMap(data); });
      // The Radar & Satellite Feeds map is NOT initialised here. Building it
      // eagerly started its RainViewer tile fetches while the user was still on
      // the Dashboard, so a network trace showed the dashboard pulling radar
      // imagery it never displays. It is now created on first visit to that
      // view (see switchNowcastView), which is also where radar belongs.
      window.StormSenseDashboardData = data;

      // Render dashboard based on active mode
      updateModeUI(window.stormSenseMode);

      // Live auto-refresh. Timers are cleared first so repeated initialisation
      // can never leave two intervals running and double up requests.
      if (liveRefreshTimer) clearInterval(liveRefreshTimer);
      if (countdownTimer) clearInterval(countdownTimer);

      resetRefreshCountdown();

      // Resolve the current live location once at startup; the panel and the
      // map marker both follow whatever this returns (including "unavailable").
      acquireLiveLocation().then(function (loc) {
        window.applyLiveLocation(loc);
      });

      // One second tick drives BOTH the visible countdown and the firing of the
      // refresh, from the same deadline -- so they cannot disagree, and a manual
      // refresh that moves the deadline also moves the auto-refresh.
      countdownTimer = setInterval(function () {
        renderCountdown();
        if (window.stormSenseMode !== "live") return;
        if (Date.now() >= nextRefreshAt) {
          // Move the deadline BEFORE awaiting, so a slow cycle cannot queue up
          // a second overlapping refresh.
          resetRefreshCountdown();
          window.refreshLiveDashboard(false);
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
        var lead = apiLeadHours();
        var nowcastData = window.StormSenseNowcastData || {};
        var issueTime = nowcastData.issue_time || "2024-05-26T12:00:00Z";
        var validTime = nowcastData.forecast_valid_time || formatUtcDateTime(new Date(parseUtcIso(issueTime).getTime() + lead * 3600 * 1000));

        var bulletin = {
          bulletin_id: "IMD-KOL-NW-" + new Date().toISOString().slice(0, 10).replace(/-/g, "") + "-" + lead + "H",
          issued_at_utc: formatUtcDateTime(parseUtcIso(issueTime)),
          forecast_valid_utc: validTime,
          lead_horizon_hours: lead,
          issuing_office: "India Meteorological Department, Regional Meteorological Centre, Kolkata",
          target_region: "Gangetic West Bengal (20.0°N–28.0°N, 84.0°E–90.0°E)",
          primary_district: "West Bengal",
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

    // Map navigation. Options come from the real boundary dataset, so every
    // entry resolves to genuine coordinates. Moving the viewport never changes
    // the monitored region, the forecast, or any displayed risk value.
    // Restore the area/horizon carried across a hard Refresh, so the button
    // behaves like Ctrl+Shift+R without dumping the user back at the default
    // view. Both values are read from the URL written by window.manualRefresh.
    loadSupportedAreas().then(function () {
      var qs = new URLSearchParams(window.location.search);
      var wantLead = qs.get("lead");
      var wantArea = qs.get("area");
      // The map must exist before jumpToArea can move it, and the horizon must
      // be applied AFTER the area so the area's pipeline run is not overwritten
      // by a whole-state one. A short defer covers initMap() completing.
      setTimeout(function () {
        if (wantArea) {
          var sel = document.getElementById("sector-selector");
          if (sel) sel.value = wantArea;
          // jumpToArea re-runs the horizon pipeline for the restored area and
          // sets StormSenseSelectedArea/SelectedDistrict.
          window.jumpToArea(wantArea);
        }
        if (wantLead != null && typeof window.setForecastHorizon === "function") {
          var n = Number(wantLead);
          if (isFinite(n)) {
            setTimeout(function () { window.setForecastHorizon(null, null, n); }, 900);
          }
        }
      }, 1200);
    }).catch(function () { return null; });

    var sectorSelect = document.getElementById("sector-selector");
    if (sectorSelect) {
      sectorSelect.addEventListener("change", function () {
        window.jumpToArea(this.value);
      });
    }
  });

  // ---------------------------------------------------------------------------
  // Jump-to-area: populated from /api/geo/areas
  // ---------------------------------------------------------------------------
  window.StormSenseAreas = null;

  function loadSupportedAreas() {
    return fetch(API_BASE + "/api/geo/areas")
      .then(parseJson)
      .then(function (data) {
        if (!data || !data.areas || !data.areas.length) return null;
        window.StormSenseAreas = data.areas;

        var sel = document.getElementById("sector-selector");
        if (!sel) return data;
        sel.innerHTML = "";
        data.areas.forEach(function (a) {
          var opt = document.createElement("option");
          opt.value = a.id;
          // An area with no model grid cells is labelled honestly rather than
          // silently behaving as though it had a forecast.
          opt.textContent = a.has_model_coverage ? a.name : a.name + " (no model coverage)";
          sel.appendChild(opt);
        });
        sel.value = "whole-state";
        return data;
      })
      .catch(function (e) {
        console.warn("Could not load supported areas:", e);
        var sel = document.getElementById("sector-selector");
        if (sel) sel.innerHTML = '<option value="whole-state">Whole State</option>';
        return null;
      });
  }

  window.jumpToArea = function (areaId) {
    if (!window.stormSenseMap) return false;
    var areas = window.StormSenseAreas || [];
    var area = null;
    for (var i = 0; i < areas.length; i++) {
      if (areas[i].id === areaId) { area = areas[i]; break; }
    }
    if (!area) {
      window.stormSenseMap.fitBounds(L.latLngBounds(window.WB_BOUNDS), { padding: [20, 20] });
      return false;
    }

    if (area.kind === "state") {
      window.stormSenseMap.fitBounds(L.latLngBounds(window.WB_BOUNDS), { padding: [20, 20] });
    } else if (area.bounds) {
      window.stormSenseMap.fitBounds(L.latLngBounds(area.bounds), { padding: [30, 30] });
    } else {
      window.stormSenseMap.flyTo(area.center, 9);
    }

    // SELECTED AREA becomes the active spatial context (CHANGE 2B).
    //
    // This used to be "a viewport change only": the map moved but every
    // dashboard card, hazard value and forecast number kept the previously
    // selected area's data, so choosing Kolkata after Murshidabad left
    // Murshidabad's numbers under a Kolkata-centred map.
    //
    // The backend already aggregates by district (/api/nowcast/summary and
    // /api/nowcast/districts both accept ?district=), so the selection is
    // recorded as state here and the existing horizon pipeline is re-run
    // against it. No values are computed in the frontend.
    //
    // NOTE: this is the SELECTED AREA, deliberately separate from the user's
    // real Current Location (window.StormSenseUserLocation), which is owned by
    // the geolocation path and must not be overwritten by an area jump.
    window.StormSenseSelectedArea = {
      id: area.id,
      name: area.name,
      kind: area.kind,
      center: area.center || null,
      hasModelCoverage: area.has_model_coverage !== false
    };
    // "whole-state" means the domain-wide peak, which is what the backend
    // returns for the default district name.
    window.StormSenseSelectedDistrict =
      (area.kind === "state") ? "West Bengal" : area.name;

    if (!area.has_model_coverage) {
      window.showToast(area.name + ": no model grid coverage for this area", "warning");
    } else {
      window.showToast("Loading " + area.name + " forecast...", "radar");
    }

    // Repaint every location-dependent component for the CURRENT horizon, so
    // location and horizon stay synchronised (CHANGE 2C).
    if (typeof window.setForecastHorizon === "function") {
      var curLead = (window.currentLeadHours === "now" || window.currentLeadHours == null)
        ? 0 : window.currentLeadHours;

      // ORDER IS LOAD-BEARING: fetch the new area's observations BEFORE
      // repainting the horizon.
      //
      // The observed-conditions strip is fed by /api/live/surface, which is
      // keyed on COORDINATES, not district, so it needs its own re-fetch for a
      // newly selected area. Previously setForecastHorizon() ran first and
      // synchronously repainted that strip from the cached
      // window.StormSenseLiveSurface -- i.e. the PREVIOUS area's payload --
      // and that paint landed after the fetch had been kicked off but before it
      // resolved. The visible result was a strip permanently one selection
      // behind: choosing Cooch Behar showed Birbhum's readings, then Bankura
      // showed Cooch Behar's, and so on. Verified against the API, which
      // returns genuinely different values per district (Bankura 81% RH /
      // 6.8 km/h vs the state default 100% / 0.0 km/h).
      //
      // Refreshing the cache first means the horizon repaint reads the area the
      // user actually selected. The fetch is defensive: if it fails, the
      // horizon still repaints rather than leaving the dashboard unresponsive.
      var repaint = function () { window.setForecastHorizon(null, null, curLead); };
      if (window.stormSenseMode === "live"
          && typeof window.fetchLiveSurfaceData === "function") {
        window.fetchLiveSurfaceData(true).then(repaint, repaint);
      } else {
        repaint();
      }
    }
    return true;
  };

  /**
   * The single authoritative district for every location-dependent request.
   *
   * Returns the user's selected area when one is active, otherwise the
   * domain-wide default. Every fetch that accepts ?district= routes through
   * this, so no component can independently invent its own spatial context.
   */
  window.activeDistrict = function () {
    return window.StormSenseSelectedDistrict || "West Bengal";
  };

  /** Query-string fragment carrying the active district, or "" for the default. */
  function districtQS() {
    var d = window.activeDistrict();
    return (d && d !== "West Bengal") ? "&district=" + encodeURIComponent(d) : "";
  }

  /**
   * Coordinates of the SELECTED AREA for point/observation endpoints.
   *
   * /api/live/surface takes lat/lon and returns the station observation nearest
   * those coordinates -- verified to differ genuinely by location (Darjeeling
   * 21.4 C vs Bankura 33.1 C at the same instant). Without this the observed
   * conditions strip stayed on the default station no matter which area was
   * selected.
   *
   * Returns "" for the whole-state selection so the backend keeps its own
   * default station, and "?" (not "&") because these endpoints take no other
   * query parameters at these call sites.
   */
  function areaCoordQS() {
    var a = window.StormSenseSelectedArea;
    if (!a || !a.center || a.kind === "state") return "";
    var lat = Number(a.center[0]), lon = Number(a.center[1]);
    if (!isFinite(lat) || !isFinite(lon)) return "";
    return "?lat=" + lat.toFixed(4) + "&lon=" + lon.toFixed(4);
  }

  // ---------------------------------------------------------------------------
  // Unified refresh control (single timer, single manual trigger)
  // ---------------------------------------------------------------------------
  var manualRefreshInFlight = false;

  window.manualRefresh = function () {
    if (manualRefreshInFlight) return Promise.resolve();
    manualRefreshInFlight = true;

    // FULL RELOAD, as requested: the Refresh button should do what
    // Ctrl+Shift+R does -- re-ingest on the server, then rebuild the whole page
    // from scratch rather than patching individual panels in place.
    //
    // This is what clears every stale frontend state variable (cached summary,
    // held observation timestamp, painted cards), which in-place repainting
    // could not fully guarantee. The selected area and horizon are carried
    // across the reload in the URL so the user lands back where they were
    // instead of being reset to the default view.
    var btnEl = document.getElementById("btn-manual-refresh");
    var iconEl = document.getElementById("manual-refresh-icon");
    if (btnEl) btnEl.disabled = true;
    if (iconEl) iconEl.classList.add("animate-spin");

    var reingest = (window.stormSenseMode === "live")
      ? fetch(API_BASE + "/api/live/refresh", { method: "POST" })
          .then(parseJson).catch(function () { return null; })
      : Promise.resolve(null);

    return reingest.then(function () {
      var params = new URLSearchParams();
      if (window.stormSenseMode === "historical") params.set("mode", "historical");
      var area = window.StormSenseSelectedArea;
      if (area && area.id && area.id !== "whole-state") params.set("area", area.id);
      var lead = window.currentLeadHours;
      params.set("lead", (lead === "now" || lead == null) ? "0" : String(lead));
      // Cache-bust so the browser cannot serve a stale document or bundle --
      // the equivalent of the hard reload this button now performs.
      params.set("_r", String(Date.now()));
      window.location.replace(
        window.location.pathname + "?" + params.toString());
    }).catch(function () {
      window.location.reload();
    });
  };

  /** Legacy in-place refresh, kept for the auto-refresh timer. */
  window.softRefresh = function () {
    if (manualRefreshInFlight) return Promise.resolve();
    manualRefreshInFlight = true;

    var btn = document.getElementById("btn-manual-refresh");
    var icon = document.getElementById("manual-refresh-icon");
    if (btn) btn.disabled = true;
    if (icon) icon.classList.add("animate-spin");

    function done() {
      manualRefreshInFlight = false;
      if (btn) btn.disabled = false;
      if (icon) icon.classList.remove("animate-spin");
      // Manual refresh restarts the countdown, per the refresh policy.
      resetRefreshCountdown();
    }

    if (window.stormSenseMode !== "live") {
      // Historical mode is a frozen case study; re-pull its products without
      // asking the backend to re-ingest live data.
      return window.setForecastHorizon(null, null, apiLeadHours())
        .then(done)
        .catch(done);
    }

    // Ask the backend to re-ingest, THEN repaint. One request, not two.
    return fetch(API_BASE + "/api/live/refresh", { method: "POST" })
      .then(parseJson)
      .catch(function (e) {
        console.warn("Backend live refresh failed:", e);
        return null;
      })
      .then(function () {
        return window.refreshLiveDashboard(true);
      })
      .then(done)
      .catch(function (e) {
        console.warn("Manual refresh failed:", e);
        done();
      });
  };
})();




