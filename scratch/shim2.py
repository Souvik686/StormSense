import os

wrapper = """
// ── Google Maps Leaflet Shim ────────────────────────────────────────────────
window.L = {
    map: function(id, options) {
        var el = typeof id === 'string' ? document.getElementById(id) : id;
        var mapOpts = {
            center: {lat: options.center[0], lng: options.center[1]},
            zoom: options.zoom,
            minZoom: options.minZoom,
            maxZoom: options.maxZoom,
            disableDefaultUI: !options.zoomControl,
            styles: [
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
            ],
            backgroundColor: '#0B0F19',
        };
        var map = new google.maps.Map(el, mapOpts);
        return {
            _gmap: map,
            on: function(event, handler) {
                if (event === 'click') {
                    map.addListener('click', function(e) {
                        handler({ latlng: { lat: e.latLng.lat(), lng: e.latLng.lng() } });
                    });
                }
            },
            setView: function(center, zoom) {
                map.setCenter({lat: center[0], lng: center[1]});
                map.setZoom(zoom);
            },
            zoomIn: function() { map.setZoom(map.getZoom() + 1); },
            zoomOut: function() { map.setZoom(map.getZoom() - 1); },
            removeLayer: function(layer) {
                if (layer && layer.setMap) { layer.setMap(null); }
            },
            fitBounds: function(bounds) {
                map.fitBounds(bounds._gbounds);
            }
        };
    },
    tileLayer: function(url, options) {
        return { addTo: function(map) { } };
    },
    imageOverlay: function(url, bounds, options) {
        var gbounds = {
            south: bounds[0][0], west: bounds[0][1],
            north: bounds[1][0], east: bounds[1][1]
        };
        var activeOverlay = new google.maps.GroundOverlay(url, gbounds, {
            opacity: options.opacity || 1.0,
            clickable: options.interactive || false
        });
        return {
            addTo: function(lMap) {
                activeOverlay.setMap(lMap._gmap);
            },
            setMap: function(m) {
                activeOverlay.setMap(m);
            },
            setUrl: function(newUrl) {
                var m = activeOverlay.getMap();
                activeOverlay.setMap(null);
                activeOverlay = new google.maps.GroundOverlay(newUrl, gbounds, {
                    opacity: options.opacity || 1.0,
                    clickable: options.interactive || false
                });
                if (m) {
                    activeOverlay.setMap(m);
                }
            },
            bringToFront: function() {},
            bringToBack: function() {}
        };
    },
    geoJSON: function(data, options) {
        var mapRef = null;
        var addedFeatures = [];
        return {
            addTo: function(lMap) {
                mapRef = lMap._gmap;
                addedFeatures = mapRef.data.addGeoJson(data);
                if (options && options.style) {
                    mapRef.data.setStyle(function(feature) {
                        var lStyle = typeof options.style === 'function' ? options.style({properties: feature.j || {}}) : options.style;
                        return {
                            fillColor: lStyle.fillColor || lStyle.color,
                            fillOpacity: lStyle.fillOpacity,
                            strokeColor: lStyle.color,
                            strokeWeight: lStyle.weight,
                            strokeOpacity: lStyle.opacity,
                            clickable: false
                        };
                    });
                }
            },
            setMap: function(m) {
                if (m === null && mapRef && addedFeatures.length > 0) {
                    for (var i = 0; i < addedFeatures.length; i++) {
                        mapRef.data.remove(addedFeatures[i]);
                    }
                }
            }
        };
    },
    marker: function(latlng, options) {
        var marker = new google.maps.Marker({
            position: {lat: latlng[0], lng: latlng[1]},
        });
        marker.addTo = function(lMap) { this.setMap(lMap._gmap); };
        marker.bindPopup = function(html) {
            var info = new google.maps.InfoWindow({ content: html });
            marker.addListener('click', function() { info.open(marker.getMap(), marker); });
            this._info = info;
            return this;
        };
        marker.openPopup = function() {
            if (this._info && this.getMap()) { this._info.open(this.getMap(), this); }
            return this;
        };
        marker.closePopup = function() {
            if (this._info) { this._info.close(); }
            return this;
        };
        return marker;
    },
    circleMarker: function(latlng, options) {
        var circle = new google.maps.Circle({
            center: {lat: latlng[0], lng: latlng[1]},
            radius: (options.radius || 10) * 1000,
            fillColor: options.fillColor || options.color,
            fillOpacity: options.fillOpacity,
            strokeColor: options.color,
            strokeWeight: options.weight,
            strokeOpacity: options.opacity
        });
        circle.addTo = function(lMap) { this.setMap(lMap._gmap); };
        circle.bindPopup = function(html) {
            var info = new google.maps.InfoWindow({ content: html });
            circle.addListener('click', function(e) {
                info.setPosition(e.latLng);
                info.open(circle.getMap());
            });
            return this;
        };
        return circle;
    },
    popup: function(options) {
        var info = new google.maps.InfoWindow();
        return {
            setLatLng: function(latlng) {
                info.setPosition({lat: latlng[0], lng: latlng[1]});
                return this;
            },
            setContent: function(content) {
                info.setContent(content);
                return this;
            },
            openOn: function(lMap) {
                info.open(lMap._gmap);
                return this;
            },
            setMap: function(m) {
                if (m === null) { info.close(); }
            }
        };
    },
    latLngBounds: function(corner1, corner2) {
        return {
            _gbounds: new google.maps.LatLngBounds(
                {lat: corner1[0], lng: corner1[1]},
                {lat: corner2[0], lng: corner2[1]}
            )
        };
    },
    divIcon: function(options) { return options; },
    layerGroup: function() {
        var layers = [];
        return {
            addLayer: function(layer) { layers.push(layer); },
            clearLayers: function() {
                layers.forEach(function(l) { if (l.setMap) l.setMap(null); });
                layers = [];
            },
            addTo: function(map) {
                layers.forEach(function(l) { if (l.addTo) l.addTo(map); });
            }
        };
    }
};
// ────────────────────────────────────────────────────────────────────────────
"""

with open('frontend/js/app.js', encoding='utf-8') as f:
    text = f.read()

# Make sure we don't duplicate it
if "Google Maps Leaflet Shim" not in text:
    text = wrapper + text

    with open('frontend/js/app.js', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Shim added")
else:
    print("Shim already exists")

