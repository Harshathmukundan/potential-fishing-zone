// api.js — Backend API calls with retry + timeout + error handling
// Uses relative URLs since frontend is served from the same Flask server
const API_BASE = '/api';

async function fetchWithTimeout(url, options = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    if (!res.ok && res.headers.get('content-type')?.includes('json')) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    return res;
  } catch (e) {
    clearTimeout(id);
    if (e.name === 'AbortError') throw new Error('Request timed out');
    throw e;
  }
}

async function fetchJSON(url, options = {}) {
  const res = await fetchWithTimeout(url, options);
  return res.json();
}

/**
 * Download a file from a URL and save it with a specific filename.
 * Uses fetch → blob → object URL → anchor click.
 * Since frontend is same-origin as API, the download attribute is respected.
 */
async function downloadFile(url, filename) {
  const res = await fetchWithTimeout(url, {}, 120000);
  const blob = await res.blob();
  const xlsxBlob = new Blob([blob], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  });
  const blobUrl = URL.createObjectURL(xlsxBlob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  }, 1000);
}

const API = {

  async status() {
    return fetchJSON(`${API_BASE}/status`);
  },

  async predict(timeIndex, minConfidence = 0.6) {
    return fetchJSON(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ time_index: timeIndex, min_confidence: minConfidence })
    });
  },

  async forecast(forecastDay) {
    return fetchJSON(`${API_BASE}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ forecast_day: forecastDay })
    });
  },

  async downloadForecastGPS(forecastDay) {
    await downloadFile(
      `${API_BASE}/forecast_gps?forecast_day=${forecastDay}`,
      `PFZ_Forecast_Day${forecastDay}.xlsx`
    );
  },

  async history() {
    return fetchJSON(`${API_BASE}/history`);
  },

  async cvResults() {
    return fetchJSON(`${API_BASE}/cv_results`);
  },

  cvPlotUrl(model) {
    return `${API_BASE}/cv_plot/${model}`;
  },

  async mapData(timeIndex) {
    return fetchJSON(`${API_BASE}/map_data`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ time_index: timeIndex })
    });
  },

  async uploadFiles(files) {
    const form = new FormData();
    files.forEach(f => form.append('files', f));
    return fetchJSON(`${API_BASE}/upload`, { method: 'POST', body: form });
  },

  gpsDownloadUrl(timeIndex, minConf = 0.6) {
    return `${API_BASE}/gps?time_index=${timeIndex}&min_confidence=${minConf}`;
  },

  async downloadGPS(timeIndex, minConf = 0.6) {
    await downloadFile(
      `${API_BASE}/gps?time_index=${timeIndex}&min_confidence=${minConf}`,
      `PFZ_GPS_t${timeIndex}.xlsx`
    );
  },

  // ── Task 1: Location-wise Forecast ──
  async locationForecast(lat, lon, locationName = '') {
    const body = locationName
      ? { location_name: locationName }
      : { lat, lon };
    return fetchJSON(`${API_BASE}/location_forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  },

  // ── Task 2+4: PFZ Ranking with Distance ──
  async pfzRankingWithDistance(lat, lon, locationName = '', minConfidence = 0.6, topN = 10) {
    return fetchJSON(`${API_BASE}/pfz_ranking_with_distance`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lat, lon,
        location_name: locationName,
        min_confidence: minConfidence,
        top_n: topN
      })
    });
  },

  // ── Task 3: Download CSV ──
  async downloadPFZCsv(lat, lon, locationName = '', minConfidence = 0.6, topN = 10) {
    const params = new URLSearchParams({
      lat, lon,
      location_name: locationName,
      min_confidence: minConfidence,
      top_n: topN
    });
    await downloadFile(
      `${API_BASE}/pfz_csv?${params.toString()}`,
      `PFZ_Top_Zones_${locationName || 'custom'}.csv`
    );
  },

  // ── User locations list ──
  async userLocations() {
    return fetchJSON(`${API_BASE}/user_locations`);
  }
};

