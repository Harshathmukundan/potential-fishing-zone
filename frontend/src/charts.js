// charts.js — All Chart.js chart definitions

const CHART_DEFAULTS = {
  color: '#8899bb',
  orange: '#f5a623',
  teal: '#00d4aa',
  red: '#e74c3c',
  blue: '#3498db',
  grid: 'rgba(30,45,69,0.8)',
};

Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.borderColor = CHART_DEFAULTS.grid;
Chart.defaults.font.family = "'Syne', sans-serif";

let historyChart = null;
let distChart = null;
let unetChart = null;
let lstmChart = null;

// ── Training history chart (dashboard) ───────────────────────
function initHistoryChart(history, metric = 'accuracy') {
  const ctx = document.getElementById('chartHistory');
  if (!ctx) return;

  if (historyChart) historyChart.destroy();

  const unet = history.unet;
  const lstm = history.convlstm;
  const len = Math.max(unet[metric].length, lstm[metric].length);
  const labels = Array.from({ length: len }, (_, i) => `Ep ${i + 1}`);

  historyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: `U-Net ${metric}`,
          data: unet[`val_${metric}`],
          borderColor: CHART_DEFAULTS.orange,
          backgroundColor: 'rgba(245,166,35,0.1)',
          borderWidth: 2.5, tension: 0.4, fill: true,
          pointRadius: 0, pointHoverRadius: 4,
        },
        {
          label: `ConvLSTM ${metric}`,
          data: lstm[`val_${metric}`],
          borderColor: CHART_DEFAULTS.teal,
          backgroundColor: 'rgba(0,212,170,0.08)',
          borderWidth: 2.5, tension: 0.4, fill: true,
          pointRadius: 0, pointHoverRadius: 4,
        },
        ...(metric === 'accuracy' ? [{
          label: '90% target',
          data: Array(len).fill(0.9),
          borderColor: 'rgba(231,76,60,0.5)',
          borderDash: [6, 4], borderWidth: 1.5,
          pointRadius: 0, fill: false,
        }] : [])
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: { backgroundColor: '#0f1829', borderColor: '#1e2d45', borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.grid }, ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: {
          grid: { color: CHART_DEFAULTS.grid },
          ticks: { font: { size: 10 } },
          ...(metric === 'accuracy' ? { min: 0.6, max: 1.0 } : {})
        }
      }
    }
  });
}

// ── PFZ distribution donut chart ─────────────────────────────
function initDistChart(low = 60, mid = 25, high = 15) {
  const ctx = document.getElementById('chartDist');
  if (!ctx) return;

  if (distChart) distChart.destroy();

  distChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Low PFZ', 'Medium PFZ', 'High PFZ'],
      datasets: [{
        data: [low, mid, high],
        backgroundColor: ['#1a3a5c', '#f39c12', '#e74c3c'],
        borderColor: ['#0f1829', '#0f1829', '#0f1829'],
        borderWidth: 3,
        hoverOffset: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 }, padding: 12 } },
        tooltip: { backgroundColor: '#0f1829', borderColor: '#1e2d45', borderWidth: 1 }
      }
    }
  });
}

// ── Full training chart (analytics panel) ────────────────────
function initFullChart(canvasId, histData, title) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const len = histData.accuracy.length;
  const labels = Array.from({ length: len }, (_, i) => `Epoch ${i + 1}`);

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Train Accuracy',
          data: histData.accuracy,
          borderColor: CHART_DEFAULTS.orange,
          borderWidth: 2, tension: 0.4, fill: false,
          pointRadius: 0,
          yAxisID: 'yAcc',
        },
        {
          label: 'Val Accuracy',
          data: histData.val_accuracy,
          borderColor: CHART_DEFAULTS.teal,
          borderWidth: 2, tension: 0.4, fill: false,
          borderDash: [5, 3], pointRadius: 0,
          yAxisID: 'yAcc',
        },
        {
          label: 'Train Loss',
          data: histData.loss,
          borderColor: CHART_DEFAULTS.red,
          borderWidth: 1.5, tension: 0.4, fill: false,
          pointRadius: 0,
          yAxisID: 'yLoss',
        },
        {
          label: 'Val Loss',
          data: histData.val_loss,
          borderColor: '#9b59b6',
          borderWidth: 1.5, tension: 0.4, fill: false,
          borderDash: [5, 3], pointRadius: 0,
          yAxisID: 'yLoss',
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { boxWidth: 10, font: { size: 11 } } },
        title: { display: true, text: title, color: CHART_DEFAULTS.color, font: { size: 13 } },
        tooltip: { backgroundColor: '#0f1829', borderColor: '#1e2d45', borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.grid }, ticks: { font: { size: 10 }, maxTicksLimit: 12 } },
        yAcc: {
          type: 'linear', position: 'left',
          grid: { color: CHART_DEFAULTS.grid },
          ticks: { font: { size: 10 } },
          min: 0.5, max: 1.0,
          title: { display: true, text: 'Accuracy', color: CHART_DEFAULTS.color }
        },
        yLoss: {
          type: 'linear', position: 'right',
          grid: { display: false },
          ticks: { font: { size: 10 } },
          title: { display: true, text: 'Loss', color: CHART_DEFAULTS.color }
        }
      }
    }
  });
}

// ── Update dist chart with prediction results ─────────────────
function updateDistChart(dist) {
  const total = dist.low + dist.medium + dist.high;
  if (distChart) {
    distChart.data.datasets[0].data = [
      +(dist.low / total * 100).toFixed(1),
      +(dist.medium / total * 100).toFixed(1),
      +(dist.high / total * 100).toFixed(1),
    ];
    distChart.update();
  }
}

// ── Per-fold accuracy bar chart (analytics) ──────────────────
function initFoldBarChart(canvasId, accData, modelName, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !accData || !accData.per_fold) return;

  const folds = accData.per_fold;
  const labels = folds.map((_, i) => `Fold ${i + 1}`);
  const meanVal = accData.mean;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: `${modelName} Accuracy`,
          data: folds.map(v => +(v * 100).toFixed(2)),
          backgroundColor: color + '99',
          borderColor: color,
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: `Mean (${(meanVal * 100).toFixed(1)}%)`,
          data: Array(folds.length).fill(+(meanVal * 100).toFixed(2)),
          type: 'line',
          borderColor: '#e74c3c',
          borderDash: [6, 4],
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: { backgroundColor: '#0f1829', borderColor: '#1e2d45', borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: CHART_DEFAULTS.grid } },
        y: {
          grid: { color: CHART_DEFAULTS.grid }, min: 80, max: 100,
          ticks: { callback: v => v + '%', font: { size: 10 } }
        }
      }
    }
  });
}

// ── Model comparison radar chart ─────────────────────────────
function initModelRadar(canvasId, unetData, lstmData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'kappa', 'miou'];
  const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', "Cohen's \u03ba", 'mIoU'];

  const unetVals = metrics.map(m => unetData[m] ? +(unetData[m].mean * 100).toFixed(1) : 0);
  const lstmVals = metrics.map(m => lstmData[m] ? +(lstmData[m].mean * 100).toFixed(1) : 0);

  new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        {
          label: 'U-Net',
          data: unetVals,
          borderColor: CHART_DEFAULTS.orange,
          backgroundColor: 'rgba(245,166,35,0.15)',
          borderWidth: 2, pointRadius: 3,
        },
        {
          label: 'ConvLSTM',
          data: lstmVals,
          borderColor: CHART_DEFAULTS.teal,
          backgroundColor: 'rgba(0,212,170,0.10)',
          borderWidth: 2, pointRadius: 3,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: { backgroundColor: '#0f1829', borderColor: '#1e2d45', borderWidth: 1 }
      },
      scales: {
        r: {
          min: 70, max: 100,
          ticks: { stepSize: 5, backdropColor: 'transparent', font: { size: 9 } },
          grid: { color: CHART_DEFAULTS.grid },
          angleLines: { color: CHART_DEFAULTS.grid },
          pointLabels: { font: { size: 10 }, color: CHART_DEFAULTS.color }
        }
      }
    }
  });
}

