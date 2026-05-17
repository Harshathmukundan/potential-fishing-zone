// map.js — Leaflet interactive map

let leafletMap     = null;
let pfzLayerGroup  = null;

const MAP_BOUNDS = [[5.0, 77.0], [14.5, 84.92]];

const PFZ_COLORS = {
  0: 'rgba(26, 58, 92, 0.65)',   // Low - dark blue
  1: 'rgba(243,156,18, 0.70)',   // Medium - amber
  2: 'rgba(231, 76, 60, 0.80)'   // High - red
};

function initMap() {
  if (leafletMap) return;

  leafletMap = L.map('leafletMap', {
    center: [9.75, 80.96],
    zoom:   7,
    minZoom: 6,
    maxZoom: 12,
    zoomControl: true,
  });

  // Dark ocean tile layer
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap contributors © CARTO',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(leafletMap);

  // Region outline
  L.rectangle(MAP_BOUNDS, {
    color:       '#f5a623',
    weight:      1.5,
    fillColor:   'transparent',
    dashArray:   '6, 4',
    opacity:     0.6,
  }).addTo(leafletMap);

  pfzLayerGroup = L.layerGroup().addTo(leafletMap);

  // Fit to study region
  leafletMap.fitBounds(MAP_BOUNDS, { padding: [20, 20] });
}

function renderPFZGrid(data) {
  if (!leafletMap) initMap();
  pfzLayerGroup.clearLayers();

  const { pred_grid, conf_grid, lat_ticks, lon_ticks } = data;
  const rows = pred_grid.length;
  const cols = pred_grid[0].length;

  const dLat = (lat_ticks[lat_ticks.length - 1] - lat_ticks[0]) / rows;
  const dLon = (lon_ticks[lon_ticks.length - 1] - lon_ticks[0]) / cols;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cls  = pred_grid[r][c];
      const conf = conf_grid[r][c];
      if (cls === 0 && conf < 0.3) continue; // skip very-low confidence low zones

      const lat1 = lat_ticks[0] + r * dLat;
      const lon1 = lon_ticks[0] + c * dLon;
      const bounds = [[lat1, lon1], [lat1 + dLat, lon1 + dLon]];

      const rect = L.rectangle(bounds, {
        color:       'transparent',
        fillColor:   PFZ_COLORS[cls],
        fillOpacity: cls === 2 ? Math.min(0.85, 0.4 + conf * 0.6) : 0.45,
        weight:      0,
      });

      if (cls === 2) {
        rect.bindTooltip(
          `<b>High PFZ</b><br>Lat: ${lat1.toFixed(3)}<br>Lon: ${lon1.toFixed(3)}<br>Conf: ${(conf*100).toFixed(1)}%`,
          { className: 'pfz-tooltip', sticky: true }
        );
      }

      pfzLayerGroup.addLayer(rect);
    }
  }
}

function addGPSMarkers(gpsPoints) {
  if (!leafletMap) return;

  const markerGroup = L.layerGroup();

  gpsPoints.slice(0, 50).forEach((pt, i) => {
    const marker = L.circleMarker([pt.lat, pt.lon], {
      radius:      i < 5 ? 8 : 5,
      fillColor:   '#e74c3c',
      color:       '#ff8080',
      weight:      1,
      fillOpacity: 0.85,
    });

    marker.bindPopup(
      `<b>High PFZ #${i + 1}</b><br>` +
      `Lat: ${pt.lat}°N<br>Lon: ${pt.lon}°E<br>` +
      `Confidence: ${(pt.confidence * 100).toFixed(1)}%`
    );

    markerGroup.addLayer(marker);
  });

  pfzLayerGroup.addLayer(markerGroup);
}
