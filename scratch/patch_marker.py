import re

with open('frontend/js/app.js', encoding='utf-8') as f:
    text = f.read()

replacement = """
    marker: function(latlng, options) {
        var markerOpts = { position: {lat: latlng[0], lng: latlng[1]} };
        if (options && options.icon && options.icon.className === 'live-location-icon') {
            markerOpts.icon = 'http://maps.google.com/mapfiles/ms/icons/purple-dot.png';
        }
        var marker = new google.maps.Marker(markerOpts);
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
"""

text = re.sub(r'    marker: function\(latlng, options\) \{.*?\n    \},\n', replacement, text, flags=re.DOTALL)

with open('frontend/js/app.js', 'w', encoding='utf-8') as f:
    f.write(text)

