// app.js — Main application controller

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let appState = {
  currentPanel: 'dashboard',
  timeIndex: 100,
  nTimes: 2766,
  minConf: 0.6,
  historyData: null,
  lastPrediction: null,
  forecastTime: 200,
};

// ─────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────
function showLoading(text = 'Processing...') {
  const el = document.getElementById('loadingOverlay');
  document.getElementById('loadingText').textContent = text;
  el.classList.add('active');
}

function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toastContainer').appendChild(el);
  setTimeout(() => {
    el.classList.add('fade-out');
    el.addEventListener('animationend', () => el.remove());
  }, 3200);
}

function formatNum(n) {
  return typeof n === 'number' ? n.toLocaleString() : n;
}

// ─────────────────────────────────────────────────────────────
// Panel navigation
// ─────────────────────────────────────────────────────────────
function switchPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const panel = document.getElementById(`panel-${name}`);
  const nav = document.querySelector(`[data-panel="${name}"]`);
  if (panel) panel.classList.add('active');
  if (nav) nav.classList.add('active');

  appState.currentPanel = name;
  document.getElementById('pageTitle').textContent =
    {
      dashboard: 'Dashboard', predict: 'Predict', forecast: 'Forecast', location: 'Location Forecast',
      analytics: 'Analytics', research: 'Research', upload: 'Upload Data'
    }[name] || name;

  // Load analytics charts
  if (name === 'analytics') loadAnalyticsPanel();
}

document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    switchPanel(item.dataset.panel);
  });
});

// ─────────────────────────────────────────────────────────────
// Startup — poll status
// ─────────────────────────────────────────────────────────────
async function initApp() {
  try {
    const s = await API.status();

    // Status dot
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    dot.className = 'status-dot online';
    text.textContent = 'API Online';

    // Pills
    document.getElementById('unetPill').textContent =
      `U-Net ${s.unet_ready ? '✓ Ready' : '○ Not loaded'}`;
    document.getElementById('lstmPill').textContent =
      `ConvLSTM ${s.lstm_ready ? '✓ Ready' : '○ Not loaded'}`;

    // Stat cards
    document.getElementById('statDays').textContent = formatNum(s.n_times || '—');

    // System info badges
    setBadge('dataStatus', s.data_loaded ? 'Loaded' : 'Not loaded', s.data_loaded ? 'badge-green' : 'badge-red');
    setBadge('unetStatus', s.unet_ready ? 'Ready' : 'Not loaded', s.unet_ready ? 'badge-green' : '');
    setBadge('lstmStatus', s.lstm_ready ? 'Ready' : 'Not loaded', s.lstm_ready ? 'badge-green' : '');

    // Update sliders max
    if (s.n_times) {
      appState.nTimes = s.n_times;
      const timeSlider = document.getElementById('timeSlider');
      if (timeSlider) timeSlider.max = s.n_times - 2;
    }

    // Load training history
    await loadHistory();
    
    // Load user locations
    loadUserLocations();

    toast('Backend connected', 'success');

  } catch (err) {
    document.getElementById('statusDot').className = 'status-dot offline';
    document.getElementById('statusText').textContent = 'API Offline';
    toast('Backend not reachable — start python app.py', 'error');
    // Load demo history anyway
    await loadHistory();
  }
}

function setBadge(id, text, cls = '') {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = `badge ${cls}`;
}

// ─────────────────────────────────────────────────────────────
// Training history
// ─────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await API.history();
    if (res.history) {
      appState.historyData = res.history;
      initHistoryChart(res.history, 'accuracy');
      initDistChart();

      // Best accuracy stats
      const uBest = Math.max(...res.history.unet.val_accuracy);
      const lBest = Math.max(...res.history.convlstm.val_accuracy);

      // Load CV Results for the Research Panel and Dashboard Stats
      await loadCVResults(uBest, lBest);

    }
  } catch (e) {
    initHistoryChart({ unet: demoHistory(), convlstm: demoHistory(0.91) });
    initDistChart();
    await loadCVResults(0.92, 0.90); // Fallback
  }
}

async function loadCVResults(fbUnet, fbLstm) {
  try {
    const res = await API.cvResults();
    if (res.cv_results && res.cv_results.unet) {
      const u = res.cv_results.unet;
      populateCVData('Unet', u);
      document.getElementById('statUnet').textContent = `${(u.accuracy.mean * 100).toFixed(1)}%`;
      document.getElementById('unetPlotImg').src = API.cvPlotUrl('unet');
    } else {
      document.getElementById('statUnet').textContent = `${(fbUnet * 100).toFixed(1)}%`;
    }

    if (res.cv_results && res.cv_results.convlstm) {
      const l = res.cv_results.convlstm;
      populateCVData('Lstm', l);
      document.getElementById('statLstm').textContent = `${(l.accuracy.mean * 100).toFixed(1)}%`;
      document.getElementById('lstmPlotImg').src = API.cvPlotUrl('convlstm');
    } else {
      document.getElementById('statLstm').textContent = `${(fbLstm * 100).toFixed(1)}%`;
    }
  } catch (e) {
    console.warn("Could not load CV results", e);
    document.getElementById('statUnet').textContent = `${(fbUnet * 100).toFixed(1)}%`;
    document.getElementById('statLstm').textContent = `${(fbLstm * 100).toFixed(1)}%`;
  }
}

function populateCVData(prefix, data) {
  // data keys are at top level: accuracy, f1_macro, kappa, miou
  const setVal = (id, val, std) => {
    const el = document.getElementById(id);
    if (el) el.textContent = `${(val * 100).toFixed(1)} ± ${(std * 100).toFixed(1)}`;
  };

  if (data.accuracy) setVal(`rs${prefix}Acc`, data.accuracy.mean, data.accuracy.std);
  if (data.f1_macro) setVal(`rs${prefix}F1`, data.f1_macro.mean, data.f1_macro.std);
  if (data.kappa) setVal(`rs${prefix}Kappa`, data.kappa.mean, data.kappa.std);
  if (data.miou) setVal(`rs${prefix}Miou`, data.miou.mean, data.miou.std);

  // per_class is an object: { "Low PFZ": { precision: { mean, std }, ... }, ... }
  const tbody = document.getElementById(`${prefix.toLowerCase()}ClassBody`);
  if (tbody && data.per_class) {
    tbody.innerHTML = Object.entries(data.per_class).map(([className, metrics]) => `
      <tr>
        <td><strong>${className}</strong></td>
        <td>${metrics.precision.mean.toFixed(3)} ± ${metrics.precision.std.toFixed(3)}</td>
        <td>${metrics.recall.mean.toFixed(3)} ± ${metrics.recall.std.toFixed(3)}</td>
        <td>${metrics.f1.mean.toFixed(3)} ± ${metrics.f1.std.toFixed(3)}</td>
        <td>${metrics.iou.mean.toFixed(3)} ± ${metrics.iou.std.toFixed(3)}</td>
      </tr>
    `).join('');
  }
}

function demoHistory(peak = 0.93) {
  const n = 18;
  const acc = Array.from({ length: n }, (_, i) =>
    +(0.65 + (peak - 0.65) * (1 - Math.exp(-0.3 * (i + 1))) + (Math.random() * 0.015 - 0.0075)).toFixed(4)
  );
  return {
    accuracy: acc.map(v => +(v - 0.02).toFixed(4)),
    val_accuracy: acc,
    loss: acc.map((_, i) => +(0.65 * Math.exp(-0.25 * i) + 0.05).toFixed(4)),
    val_loss: acc.map((_, i) => +(0.70 * Math.exp(-0.22 * i) + 0.07).toFixed(4)),
  };
}

// ── Chart mode toggle ─────────────────────────────────────────
document.querySelectorAll('[data-chart]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-chart]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (appState.historyData) initHistoryChart(appState.historyData, btn.dataset.chart);
  });
});

// ─────────────────────────────────────────────────────────────
// Predict panel
// ─────────────────────────────────────────────────────────────
const timeSlider = document.getElementById('timeSlider');
const confSlider = document.getElementById('confSlider');

timeSlider.addEventListener('input', () => {
  appState.timeIndex = parseInt(timeSlider.value);
  document.getElementById('sliderVal').textContent = timeSlider.value;
});

confSlider.addEventListener('input', () => {
  appState.minConf = parseInt(confSlider.value) / 100;
  document.getElementById('confVal').textContent = `${confSlider.value}%`;
});

document.getElementById('runPredictBtn').addEventListener('click', async () => {
  showLoading('Running U-Net prediction...');
  try {
    const res = await API.predict(appState.timeIndex, appState.minConf);
    hideLoading();

    if (res.error) { toast(res.error, 'error'); return; }

    appState.lastPrediction = res;

    // Show map image and date
    const wrap = document.getElementById('predictMapWrap');
    wrap.innerHTML = `
      <h4 style="color:var(--orange); text-align:center; margin-bottom: 15px; font-size: 16px; font-weight: 600;">
        Prediction Date: ${res.date_str}
      </h4>
      <img style="max-width:100%; height:auto; border-radius: 8px;" src="data:image/png;base64,${res.map_image}" alt="PFZ Map" />
    `;

    // Stats (convert to percentages)
    const stats = document.getElementById('predictStats');
    stats.style.display = 'block';
    const totalPx = res.distribution.low + res.distribution.medium + res.distribution.high;
    const calcPct = (val) => ((val / totalPx) * 100).toFixed(1) + '%';

    document.getElementById('lowCount').textContent = calcPct(res.distribution.low);
    document.getElementById('midCount').textContent = calcPct(res.distribution.medium);
    document.getElementById('highCount').textContent = calcPct(res.distribution.high);

    // Model badge
    document.getElementById('modelUsedBadge').textContent = res.model_used;

    // GPS table
    if (res.gps_top20 && res.gps_top20.length > 0) {
      populateGPSTable(res.gps_top20);
      document.getElementById('gpsCard').style.display = 'block';
    }

    // Export button
    document.getElementById('exportGpsBtn').style.display = 'flex';

    // Update dist chart
    updateDistChart(res.distribution);

    toast(`Prediction complete — ${res.gps_count} high PFZ pixels found`, 'success');

  } catch (e) {
    hideLoading();
    toast('Prediction failed — is backend running?', 'error');
  }
});

document.getElementById('exportGpsBtn').addEventListener('click', async () => {
  try {
    await API.downloadGPS(appState.timeIndex, appState.minConf);
    toast('GPS Excel downloaded!', 'success');
  } catch (e) {
    toast('Export failed', 'error');
  }
});

function populateGPSTable(points) {
  const tbody = document.getElementById('gpsTableBody');
  tbody.innerHTML = points.map((p, i) => {
    const conf = p.confidence;
    const cls = conf > 0.8 ? 'conf-high' : conf > 0.65 ? 'conf-mid' : 'conf-low';
    return `<tr>
      <td>${i + 1}</td>
      <td>${p.lat.toFixed(4)}</td>
      <td>${p.lon.toFixed(4)}</td>
      <td class="${cls}">${(conf * 100).toFixed(1)}%</td>
    </tr>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────
// Forecast panel (single-day with GPS)
// ─────────────────────────────────────────────────────────────

// Generate 7 day-selector buttons
const FORECAST_DATES = [
  { day: 1, label: 'Today', date: '2026-03-27' },
  { day: 2, label: 'Day 2', date: '2026-03-28' },
  { day: 3, label: 'Day 3', date: '2026-03-29' },
  { day: 4, label: 'Day 4', date: '2026-03-30' },
  { day: 5, label: 'Day 5', date: '2026-03-31' },
  { day: 6, label: 'Day 6', date: '2026-04-01' },
  { day: 7, label: 'Day 7', date: '2026-04-02' },
];

const forecastGrid = document.getElementById('forecastDayGrid');
FORECAST_DATES.forEach(fd => {
  const btn = document.createElement('button');
  btn.className = 'forecast-day-btn';
  btn.dataset.day = fd.day;
  btn.innerHTML = `<span class="fd-label">${fd.label}</span><span class="fd-date">${fd.date}</span>`;
  btn.addEventListener('click', () => runForecastForDay(fd.day, btn));
  forecastGrid.appendChild(btn);
});

appState.activeForecastDay = null;

async function runForecastForDay(forecastDay, btnEl) {
  // Highlight active button
  document.querySelectorAll('.forecast-day-btn').forEach(b => b.classList.remove('active'));
  btnEl.classList.add('active');
  appState.activeForecastDay = forecastDay;

  showLoading(`Running ConvLSTM forecast for ${FORECAST_DATES[forecastDay - 1].date}...`);
  try {
    const res = await API.forecast(forecastDay);
    hideLoading();

    if (res.error) { toast(res.error, 'error'); return; }

    // Show map
    const wrap = document.getElementById('forecastMapWrap');
    wrap.innerHTML = `
      <h4 style="color:var(--orange); text-align:center; margin-bottom: 15px; font-size: 16px; font-weight: 600;">
        Forecast Date: ${res.date_str}
      </h4>
      <img style="max-width:100%; height:auto; border-radius: 8px;" src="data:image/png;base64,${res.forecast_img}" alt="Forecast Map" />
    `;

    // Stats
    const stats = document.getElementById('forecastStats');
    stats.style.display = 'block';
    const totalPx = res.distribution.low + res.distribution.medium + res.distribution.high;
    const calcPct = (val) => ((val / totalPx) * 100).toFixed(1) + '%';
    document.getElementById('fcLowCount').textContent = calcPct(res.distribution.low);
    document.getElementById('fcMidCount').textContent = calcPct(res.distribution.medium);
    document.getElementById('fcHighCount').textContent = calcPct(res.distribution.high);
    document.getElementById('fcModelBadge').textContent = res.model_used;

    // GPS table
    if (res.gps_top20 && res.gps_top20.length > 0) {
      const tbody = document.getElementById('fcGpsTableBody');
      tbody.innerHTML = res.gps_top20.map((p, i) => {
        const conf = p.confidence;
        const cls = conf > 0.8 ? 'conf-high' : conf > 0.65 ? 'conf-mid' : 'conf-low';
        return `<tr>
          <td>${i + 1}</td>
          <td>${p.lat.toFixed(4)}</td>
          <td>${p.lon.toFixed(4)}</td>
          <td class="${cls}">${(conf * 100).toFixed(1)}%</td>
        </tr>`;
      }).join('');
      document.getElementById('fcGpsCard').style.display = 'block';
    }

    // Export button
    document.getElementById('exportForecastGpsBtn').style.display = 'flex';

    toast(`Forecast complete — ${res.gps_count} high PFZ pixels found for ${res.date_str}`, 'success');
  } catch (e) {
    hideLoading();
    toast('Forecast failed — is backend running?', 'error');
  }
}

document.getElementById('exportForecastGpsBtn').addEventListener('click', async () => {
  if (!appState.activeForecastDay) return;
  try {
    await API.downloadForecastGPS(appState.activeForecastDay);
    toast('Forecast GPS Excel downloaded!', 'success');
  } catch (e) {
    toast('Export failed', 'error');
  }
});

// ─────────────────────────────────────────────────────────────────
// Analytics panel
// ─────────────────────────────────────────────────────────────────
let analyticsLoaded = false;

async function loadAnalyticsPanel() {
  // Training history charts
  if (appState.historyData) {
    if (!unetChart) unetChart = initFullChart('chartUnetFull', appState.historyData.unet, 'U-Net Training History');
    if (!lstmChart) lstmChart = initFullChart('chartLstmFull', appState.historyData.convlstm, 'ConvLSTM Training History');
  }

  if (analyticsLoaded) return;

  // Load CV results for per-fold charts and confusion matrices
  try {
    const res = await API.cvResults();
    if (!res.cv_results) return;

    const cv = res.cv_results;

    // Per-fold accuracy bar charts (data.accuracy.per_fold)
    if (cv.unet && cv.unet.accuracy) {
      initFoldBarChart('chartUnetFolds', cv.unet.accuracy, 'U-Net', CHART_DEFAULTS.orange);
    }
    if (cv.convlstm && cv.convlstm.accuracy) {
      initFoldBarChart('chartLstmFolds', cv.convlstm.accuracy, 'ConvLSTM', CHART_DEFAULTS.teal);
    }

    // Model comparison radar chart
    if (cv.unet && cv.convlstm) {
      initModelRadar('chartModelRadar', cv.unet, cv.convlstm);
    }

    // Confusion matrices (aggregated_confusion_matrix — raw counts, need to normalize)
    if (cv.unet && cv.unet.aggregated_confusion_matrix) {
      populateCM('unetCMBody', normalizeCM(cv.unet.aggregated_confusion_matrix));
    }
    if (cv.convlstm && cv.convlstm.aggregated_confusion_matrix) {
      populateCM('lstmCMBody', normalizeCM(cv.convlstm.aggregated_confusion_matrix));
    }

    analyticsLoaded = true;
  } catch (e) {
    console.warn('Could not load CV results for analytics', e);
  }
}

function normalizeCM(cm) {
  // Convert raw counts to row-normalized percentages
  return cm.map(row => {
    const total = row.reduce((a, b) => a + b, 0);
    return total > 0 ? row.map(v => v / total) : row;
  });
}

function populateCM(tbodyId, cm) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody || !cm) return;
  const labels = ['Low', 'Medium', 'High'];
  tbody.innerHTML = cm.map((row, i) => {
    const cells = row.map((val, j) => {
      const pct = (val * 100).toFixed(1);
      const isDiag = (i === j);
      return `<td style="${isDiag ? 'font-weight:700; color:var(--orange);' : ''}">${pct}%</td>`;
    }).join('');
    return `<tr><td><strong>${labels[i]}</strong></td>${cells}</tr>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────
// Upload panel
// ─────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const uploadBtn = document.getElementById('uploadBtn');
let selectedFiles = [];

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  handleFiles(Array.from(e.dataTransfer.files));
});

fileInput.addEventListener('change', () => handleFiles(Array.from(fileInput.files)));

function handleFiles(files) {
  selectedFiles = files.filter(f => f.name.endsWith('.nc'));
  fileList.innerHTML = selectedFiles.map(f =>
    `<div class="file-item">
      <span>${f.name}</span>
      <span class="file-size">${(f.size / 1e6).toFixed(1)} MB</span>
    </div>`
  ).join('');

  if (selectedFiles.length > 0) {
    uploadBtn.style.display = 'flex';
    toast(`${selectedFiles.length} file(s) selected`, 'info');
  }
}

uploadBtn.addEventListener('click', async () => {
  if (selectedFiles.length === 0) return;
  showLoading('Uploading NetCDF files...');
  try {
    const res = await API.uploadFiles(selectedFiles);
    hideLoading();

    if (res.success) {
      toast(`${res.count} file(s) uploaded successfully`, 'success');
      fileList.innerHTML = '';
      uploadBtn.style.display = 'none';
      selectedFiles = [];
      await initApp(); // refresh status
    } else {
      toast(res.error || 'Upload failed', 'error');
    }
  } catch (e) {
    hideLoading();
    toast('Upload failed', 'error');
  }
});

// ─────────────────────────────────────────────────────────────
// Location Forecast Panel
// ─────────────────────────────────────────────────────────────

const locSelect = document.getElementById('locSelect');
const locLat = document.getElementById('locLat');
const locLon = document.getElementById('locLon');
const locConfSlider = document.getElementById('locConfSlider');

let currentLocName = '';
let currentMinConf = 0.6;

async function loadUserLocations() {
  try {
    const res = await API.userLocations();
    if (res.success && res.locations) {
      res.locations.forEach(loc => {
        const opt = document.createElement('option');
        opt.value = JSON.stringify({ lat: loc.lat, lon: loc.lon, name: loc.name });
        opt.textContent = loc.name;
        locSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn('Failed to load user locations', e);
  }
}

locSelect.addEventListener('change', () => {
  if (locSelect.value) {
    const data = JSON.parse(locSelect.value);
    locLat.value = data.lat;
    locLon.value = data.lon;
    currentLocName = data.name;
  } else {
    currentLocName = '';
  }
});

locLat.addEventListener('input', () => { locSelect.value = ''; currentLocName = ''; });
locLon.addEventListener('input', () => { locSelect.value = ''; currentLocName = ''; });

locConfSlider.addEventListener('input', () => {
  currentMinConf = parseInt(locConfSlider.value) / 100;
  document.getElementById('locConfVal').textContent = `${locConfSlider.value}%`;
});

document.getElementById('runLocForecastBtn').addEventListener('click', async () => {
  const lat = parseFloat(locLat.value);
  const lon = parseFloat(locLon.value);
  if (isNaN(lat) || isNaN(lon)) {
    toast('Please enter valid coordinates', 'error');
    return;
  }

  showLoading('Fetching location forecast...');
  try {
    const res = await API.locationForecast(lat, lon, currentLocName);
    hideLoading();

    if (res.error) { toast(res.error, 'error'); return; }

    const tbody = document.getElementById('locForecastBody');
    tbody.innerHTML = res.predictions.map(p => {
      const cls = p.pfz_class === 2 ? 'conf-high' : p.pfz_class === 1 ? 'conf-mid' : 'conf-low';
      return `<tr>
        <td>Day ${p.day}</td>
        <td>${p.date}</td>
        <td class="${cls}"><strong>${p.pfz_label}</strong></td>
        <td>${p.confidence.toFixed(1)}%</td>
        <td>${p.is_land ? 'Yes' : 'No'}</td>
      </tr>`;
    }).join('');

    document.getElementById('locForecastCard').style.display = 'block';
    toast(`Forecast loaded for ${res.location_name}`, 'success');
  } catch (e) {
    hideLoading();
    toast('Forecast failed', 'error');
  }
});

document.getElementById('runLocRankingBtn').addEventListener('click', async () => {
  const lat = parseFloat(locLat.value);
  const lon = parseFloat(locLon.value);
  if (isNaN(lat) || isNaN(lon)) {
    toast('Please enter valid coordinates', 'error');
    return;
  }

  showLoading('Finding nearby PFZ zones...');
  try {
    const res = await API.pfzRankingWithDistance(lat, lon, currentLocName, currentMinConf, 10);
    hideLoading();

    if (res.error) { toast(res.error, 'error'); return; }

    const container = document.getElementById('locRankingContainer');
    container.innerHTML = res.rankings.map(dayRank => {
      if (dayRank.top_zones.length === 0) {
        return `<div><strong style="color:var(--orange)">Day ${dayRank.day} (${dayRank.date}):</strong> No high PFZ zones found nearby.</div>`;
      }

      const rows = dayRank.top_zones.map((z, i) => `
        <tr>
          <td>${i + 1}</td>
          <td>${z.lat.toFixed(4)}</td>
          <td>${z.lon.toFixed(4)}</td>
          <td class="conf-high">${z.confidence.toFixed(1)}%</td>
          <td>${z.distance_km} km</td>
          <td>${z.bearing}</td>
        </tr>
      `).join('');

      return `
        <div style="margin-bottom: 10px;">
          <strong style="color:var(--orange)">Day ${dayRank.day} (${dayRank.date})</strong> — Found ${dayRank.total_high} zones
          <div class="table-wrap" style="margin-top: 5px;">
            <table class="data-table">
              <thead>
                <tr><th>#</th><th>Lat</th><th>Lon</th><th>Conf</th><th>Dist</th><th>Dir</th></tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      `;
    }).join('');

    document.getElementById('locRankingCard').style.display = 'block';
    document.getElementById('exportLocGpsBtn').style.display = 'flex';
    toast('Nearby zones loaded', 'success');
  } catch (e) {
    hideLoading();
    toast('Ranking failed', 'error');
  }
});

document.getElementById('exportLocGpsBtn').addEventListener('click', async () => {
  const lat = parseFloat(locLat.value);
  const lon = parseFloat(locLon.value);
  try {
    await API.downloadPFZCsv(lat, lon, currentLocName, currentMinConf, 10);
    toast('CSV Downloaded!', 'success');
  } catch (e) {
    toast('Download failed', 'error');
  }
});

// ─────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});
