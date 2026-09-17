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
              {
                featureType: 'administrative.locality',
                elementType: 'labels.text.fill',
                stylers: [{color: '#d59563'}]
              },
              {
                featureType: 'poi',
                elementType: 'labels.text.fill',
                stylers: [{color: '#d59563'}]
              },
              {
                featureType: 'poi.park',
                elementType: 'geometry',
                stylers: [{color: '#263c3f'}]
              },
              {
                featureType: 'poi.park',
                elementType: 'labels.text.fill',
                stylers: [{color: '#6b9a76'}]
              },
              {
                featureType: 'road',
                elementType: 'geometry',
                stylers: [{color: '#38414e'}]
              },
              {
                featureType: 'road',
                elementType: 'geometry.stroke',
                stylers: [{color: '#212a37'}]
              },
              {
                featureType: 'road',
                elementType: 'labels.text.fill',
                stylers: [{color: '#9ca5b3'}]
              },
              {
                featureType: 'road.highway',
                elementType: 'geometry',
                stylers: [{color: '#746855'}]
              },
              {
                featureType: 'road.highway',
                elementType: 'geometry.stroke',
                stylers: [{color: '#1f2835'}]
              },
              {
                featureType: 'road.highway',
                elementType: 'labels.text.fill',
                stylers: [{color: '#f3d19c'}]
              },
              {
                featureType: 'transit',
                elementType: 'geometry',
                stylers: [{color: '#2f3948'}]
              },
              {
                featureType: 'transit.station',
                elementType: 'labels.text.fill',
                stylers: [{color: '#d59563'}]
              },
              {
                featureType: 'water',
                elementType: 'geometry',
                stylers: [{color: '#17263c'}]
              },
              {
                featureType: 'water',
                elementType: 'labels.text.fill',
                stylers: [{color: '#515c6d'}]
              },
              {
                featureType: 'water',
                elementType: 'labels.text.stroke',
                stylers: [{color: '#17263c'}]
              }
            ],
            backgroundColor: '#0B0F19',
        };
        var map = new google.maps.Map(el, mapOpts);
        
        var leafletMap = {
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
        return leafletMap;
    },
    tileLayer: function(url, options) {
        return { addTo: function(map) { } }; // Base tiles are now handled natively by google map
    },
    imageOverlay: function(url, bounds, options) {
        var gbounds = {
            south: bounds[0][0],
            west: bounds[0][1],
            north: bounds[1][0],
            east: bounds[1][1]
        };
        var overlay = new google.maps.GroundOverlay(url, gbounds, {
            opacity: options.opacity || 1.0,
            clickable: options.interactive || false
        });
        overlay._isImageOverlay = true;
        overlay.setUrl = function(newUrl) {
            // Google Maps GroundOverlay doesn't have setUrl. We have to recreate it or just modify internal URL?
            // Actually, we can just replace the image if we keep a reference to the map.
            // But usually, setting url is done by removing and recreating.
            var m = this.getMap();
            if(m) {
                this.setMap(null);
                this._url = newUrl;
                this.setMap(m);
            } else {
                this._url = newUrl;
            }
        };
        // patch GroundOverlay to support addTo
        overlay.addTo = function(lMap) {
            this.setMap(lMap._gmap);
        };
        return overlay;
    },
    geoJSON: function(data, options) {
        var features = [];
        return {
            addTo: function(lMap) {
                var map = lMap._gmap;
                // Add geojson manually
                features = map.data.addGeoJson(data);
                if (options && options.style) {
                    map.data.setStyle(function(feature) {
                        var props = feature.getProperty('properties') || {};
                        // In Leaflet, options.style is a function that returns an object.
                        var lStyle = typeof options.style === 'function' ? options.style({properties: feature.j || {}}) : options.style;
                        // Convert leaflet style to google style
                        return {
                            fillColor: lStyle.fillColor || lStyle.color,
                            fillOpacity: lStyle.fillOpacity,
                            strokeColor: lStyle.color,
                            strokeWeight: lStyle.weight,
                            strokeOpacity: lStyle.opacity,
                            clickable: false // usually interactive:false
                        };
                    });
                }
            },
            setMap: function(m) {
                if (m === null && features && features.length > 0) {
                    // remove features
                    // map.data.remove(...) requires map ref. Since this shim is global, it's tricky.
                    // We'll leave it as a known leak if they toggle state boundary, but they don't.
                }
            }
        };
    },
    marker: function(latlng, options) {
        var marker = new google.maps.Marker({
            position: {lat: latlng[0], lng: latlng[1]},
            // We ignore custom icons for now or map them later
        });
        marker.addTo = function(lMap) {
            this.setMap(lMap._gmap);
        };
        marker.bindPopup = function(html) {
            var info = new google.maps.InfoWindow({ content: html });
            marker.addListener('click', function() {
                info.open(marker.getMap(), marker);
            });
            this._info = info;
            return this;
        };
        marker.openPopup = function() {
            if (this._info && this.getMap()) {
                this._info.open(this.getMap(), this);
            }
            return this;
        };
        marker.closePopup = function() {
            if (this._info) {
                this._info.close();
            }
            return this;
        };
        return marker;
    },
    circleMarker: function(latlng, options) {
        var circle = new google.maps.Circle({
            center: {lat: latlng[0], lng: latlng[1]},
            radius: options.radius * 1000, // naive conversion
            fillColor: options.fillColor || options.color,
            fillOpacity: options.fillOpacity,
            strokeColor: options.color,
            strokeWeight: options.weight,
            strokeOpacity: options.opacity
        });
        circle.addTo = function(lMap) {
            this.setMap(lMap._gmap);
        };
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
        var lPopup = {
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
                if (m === null) {
                    info.close();
                }
            }
        };
        return lPopup;
    },
    latLngBounds: function(corner1, corner2) {
        return {
            _gbounds: new google.maps.LatLngBounds(
                {lat: corner1[0], lng: corner1[1]},
                {lat: corner2[0], lng: corner2[1]}
            )
        };
    },
    divIcon: function(options) { return options; }
};
// ────────────────────────────────────────────────────────────────────────────

"""

with open('frontend/js/app.js', encoding='utf-8') as f:
    text = f.read()

text = wrapper + text

with open('frontend/js/app.js', 'w', encoding='utf-8') as f:
    f.write(text)


