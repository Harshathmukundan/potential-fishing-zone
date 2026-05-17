"""
app.py — Flask REST API Backend
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N

Run:
    python app.py
    python app.py --data_dir /path/to/nc_files --port 5001
"""

import argparse
import os
import io
import base64
import json
import math
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import model_handler  as mh
import data_processor as dp

# ─── App setup ───────────────────────────────────────────────
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app      = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

UPLOAD_DIR   = os.path.join(os.path.dirname(__file__), 'uploads')
_data_subdir = os.path.join(UPLOAD_DIR, 'DATA')
DATA_DIR     = os.environ.get('PFZ_DATA_DIR',
                               _data_subdir if os.path.isdir(_data_subdir) else UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models'), exist_ok=True)

_das     = None
_N_TIMES = None
_loaded  = False


def get_data():
    global _das, _N_TIMES, _loaded
    if not _loaded:
        try:
            _das, _N_TIMES = dp.load_data(DATA_DIR)
            _loaded = True
        except Exception as e:
            print(f"⚠️  Data not yet loaded: {e}")
    return _das, _N_TIMES


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


# ─── Health & Status ─────────────────────────────────────────

@app.route('/api/status', methods=['GET'])
def status():
    das, N = get_data()
    return jsonify({
        'status':       'online',
        'data_loaded':  das is not None,
        'n_times':      int(N) if N else 0,
        'unet_ready':   mh.is_unet_ready(),
        'lstm_ready':   mh.is_lstm_ready(),
        'model_info':   mh.get_model_info(),
        'data_dir':     DATA_DIR
    })


# ─── Predict endpoint ────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run U-Net PFZ prediction for a given time index.
    Body JSON: { "time_index": int, "min_confidence": float }
    """
    try:
        body       = request.get_json(silent=True) or {}
        t          = int(body.get('time_index', 0))
        min_conf   = float(body.get('min_confidence', 0.6))

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded. Upload NetCDF files first.'}), 400

        n_times = int(N) if N is not None else 0
        t = int(np.clip(t, 1, n_times - 2))

        # Build features
        temporal = dp.build_temporal_sample(das, t, N)
        single   = dp.build_sample(das, t)

        # Run model or use domain labels if model not loaded
        if mh.is_unet_ready():
            prob_map = mh.predict_unet(temporal)
        else:
            label    = dp.make_label(single)
            prob_map = np.zeros((256, 256, 3), dtype=np.float32)
            for cls in range(3):
                prob_map[:,:,cls] = np.where(label == cls, 1.0, 0.0).astype(np.float32)

        pred_map  = np.argmax(prob_map, axis=-1)
        conf_map  = prob_map[:,:,2]  # High PFZ confidence

        land_mask = dp.get_land_mask()
        pred_map[land_mask] = 0
        conf_map[land_mask] = 0.0

        # Class distribution (excluding land)
        ocean_pred = pred_map[~land_mask]
        dist  = {
            'low':    int(np.sum(ocean_pred == 0)),
            'medium': int(np.sum(ocean_pred == 1)),
            'high':   int(np.sum(ocean_pred == 2))
        }

        # GPS export
        gps_points = dp.pfz_to_gps(pred_map, conf_map, min_confidence=min_conf)

        # Channel stats
        stats = dp.get_channel_stats(das, t)

        # Generate PFZ map image
        pm_plot = pred_map.astype(float)
        pm_plot[land_mask] = np.nan
        cm_plot = conf_map.copy()
        cm_plot[land_mask] = np.nan

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                  facecolor='#0a0e1a')
        ax0, ax1 = axes[0], axes[1] # type: ignore

        for ax in [ax0, ax1]:
            ax.set_facecolor('#000000')

        cmap_pfz = mcolors.ListedColormap(['#1a3a5c', '#f39c12', '#e74c3c'])
        cmap_pfz.set_bad('#000000') # Land pixels
        im0 = ax0.imshow(pm_plot, cmap=cmap_pfz, origin='lower',
                              extent=(float(dp.LON_MIN), float(dp.LON_MAX), float(dp.LAT_MIN), float(dp.LAT_MAX)),
                              vmin=0, vmax=2)
        ax0.set_title('PFZ Classification Map', color='white', fontsize=12, fontweight='bold')
        ax0.set_xlabel('Longitude (°E)', color='#8899aa')
        ax0.set_ylabel('Latitude (°N)',  color='#8899aa')
        ax0.tick_params(colors='#8899aa')
        cbar0 = plt.colorbar(im0, ax=ax0, ticks=[0,1,2], shrink=0.8)
        cbar0.ax.set_yticklabels(['Low','Medium','High'], color='white')
        cbar0.ax.tick_params(colors='white')

        cmap_conf = plt.cm.YlOrRd.copy()
        cmap_conf.set_bad('#000000')
        im1 = ax1.imshow(cm_plot, cmap=cmap_conf, origin='lower',
                              extent=(float(dp.LON_MIN), float(dp.LON_MAX), float(dp.LAT_MIN), float(dp.LAT_MAX)))
        ax1.set_title('High PFZ Confidence Score', color='white', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Longitude (°E)', color='#8899aa')
        ax1.set_ylabel('Latitude (°N)',  color='#8899aa')
        ax1.tick_params(colors='#8899aa')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.ax.tick_params(colors='white')

        plt.tight_layout()
        map_img = fig_to_base64(fig)

        return jsonify({
            'success':     True,
            'time_index':  t,
            'date_str':    dp.get_date_for_index(das, t),
            'distribution': dist,
            'gps_count':   len(gps_points),
            'gps_top20':   gps_points[:20],
            'stats':       stats,
            'map_image':   map_img,
            'model_used':  'U-Net' if mh.is_unet_ready() else 'Domain Rules'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Forecast endpoint (single-day with GPS) ─────────────────

# Mocked "today" dates for presentation: 2026-03-27 to 2026-04-02
from datetime import date, timedelta
FORECAST_BASE_DATE = date(2026, 3, 27)

def _forecast_single_day(das, N, forecast_day, n_times):
    """Run ConvLSTM for a single forecast day (1-7) and return results."""
    land_mask = dp.get_land_mask()

    # Use a fixed base time index deep in the dataset so models have enough history
    base_t = min(1000, n_times - 10)
    target_t = int(np.clip(base_t + forecast_day, 7, n_times - 1))

    # Build sequence ending at target_t - 1 to predict target_t
    seq = dp.build_lstm_sequence(das, target_t - 1, N)

    if mh.is_lstm_ready():
        prob_map = mh.predict_convlstm(seq)
    else:
        single = dp.build_sample(das, target_t)
        label = dp.make_label(single)
        prob_map = np.zeros((256, 256, 3), dtype=np.float32)
        for cls in range(3):
            prob_map[:,:,cls] = np.where(label == cls, 1.0, 0.0).astype(np.float32)

    pred_map = np.argmax(prob_map, axis=-1)
    conf_map = prob_map[:,:,2]

    pred_map[land_mask] = 0
    conf_map[land_mask] = 0.0

    return pred_map, conf_map, land_mask, target_t


@app.route('/api/forecast', methods=['POST'])
def forecast():
    """
    Single-day PFZ forecast using ConvLSTM.
    Body JSON: { "forecast_day": 1-7 }
    forecast_day 1 = today (2026-03-27), 2 = tomorrow, ... 7 = 2026-04-02
    """
    try:
        body = request.get_json(silent=True) or {}
        forecast_day = int(body.get('forecast_day', 1))
        forecast_day = max(1, min(7, forecast_day))

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0

        pred_map, conf_map, land_mask, target_t = _forecast_single_day(das, N, forecast_day, n_times)

        # Mocked presentation date
        display_date = FORECAST_BASE_DATE + timedelta(days=forecast_day - 1)
        date_str = display_date.strftime('%Y-%m-%d')

        # Class distribution (excluding land)
        ocean_pred = pred_map[~land_mask]
        dist = {
            'low':    int(np.sum(ocean_pred == 0)),
            'medium': int(np.sum(ocean_pred == 1)),
            'high':   int(np.sum(ocean_pred == 2))
        }

        # GPS points
        gps_points = dp.pfz_to_gps(pred_map, conf_map, min_confidence=0.5)

        # Generate single-day map image
        pm_plot = pred_map.astype(float)
        pm_plot[land_mask] = np.nan
        cm_plot = conf_map.copy()
        cm_plot[land_mask] = np.nan

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0a0e1a')
        ax0, ax1 = axes[0], axes[1]  # type: ignore
        for ax in [ax0, ax1]:
            ax.set_facecolor('#000000')

        cmap_pfz = mcolors.ListedColormap(['#1a3a5c', '#f39c12', '#e74c3c'])
        cmap_pfz.set_bad('#000000')
        im0 = ax0.imshow(pm_plot, cmap=cmap_pfz, origin='lower',
                         extent=(float(dp.LON_MIN), float(dp.LON_MAX), float(dp.LAT_MIN), float(dp.LAT_MAX)),
                         vmin=0, vmax=2)
        ax0.set_title(f'PFZ Forecast — {date_str}', color='white', fontsize=12, fontweight='bold')
        ax0.set_xlabel('Longitude (°E)', color='#8899aa')
        ax0.set_ylabel('Latitude (°N)', color='#8899aa')
        ax0.tick_params(colors='#8899aa')
        cbar0 = plt.colorbar(im0, ax=ax0, ticks=[0,1,2], shrink=0.8)
        cbar0.ax.set_yticklabels(['Low','Medium','High'], color='white')
        cbar0.ax.tick_params(colors='white')

        cmap_conf = plt.cm.YlOrRd.copy()
        cmap_conf.set_bad('#000000')
        im1 = ax1.imshow(cm_plot, cmap=cmap_conf, origin='lower',
                         extent=(float(dp.LON_MIN), float(dp.LON_MAX), float(dp.LAT_MIN), float(dp.LAT_MAX)))
        ax1.set_title('High PFZ Confidence Score', color='white', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Longitude (°E)', color='#8899aa')
        ax1.set_ylabel('Latitude (°N)', color='#8899aa')
        ax1.tick_params(colors='#8899aa')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.ax.tick_params(colors='white')

        plt.tight_layout()
        forecast_img = fig_to_base64(fig)

        return jsonify({
            'success':      True,
            'forecast_day': forecast_day,
            'date_str':     date_str,
            'distribution': dist,
            'gps_count':    len(gps_points),
            'gps_top20':    gps_points[:20],
            'forecast_img': forecast_img,
            'model_used':   'ConvLSTM' if mh.is_lstm_ready() else 'Domain Rules'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Forecast GPS Excel Export ───────────────────────────────

@app.route('/api/forecast_gps', methods=['GET', 'POST'])
def export_forecast_gps():
    """Export forecast GPS coordinates as Excel download."""
    try:
        if request.method == 'GET':
            forecast_day = int(request.args.get('forecast_day', 1))
        else:
            body = request.get_json(silent=True) or {}
            forecast_day = int(body.get('forecast_day', 1))
        forecast_day = max(1, min(7, forecast_day))

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        pred_map, conf_map, land_mask, target_t = _forecast_single_day(das, N, forecast_day, n_times)

        gps = dp.pfz_to_gps(pred_map, conf_map, min_confidence=0.5)

        display_date = FORECAST_BASE_DATE + timedelta(days=forecast_day - 1)
        date_str = display_date.strftime('%Y-%m-%d')

        df = pd.DataFrame(gps)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)

        return send_file(
            buf,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'PFZ_Forecast_{date_str}.xlsx'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── GPS Excel Export ────────────────────────────────────────

@app.route('/api/gps', methods=['GET', 'POST'])
def export_gps():
    """Export GPS coordinates as Excel download."""
    try:
        if request.method == 'GET':
            t        = int(request.args.get('time_index', 0))
            min_conf = float(request.args.get('min_confidence', 0.6))
        else:
            body     = request.get_json(silent=True) or {}
            t        = int(body.get('time_index', 0))
            min_conf = float(body.get('min_confidence', 0.6))

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        t       = int(np.clip(t, 1, n_times - 2))
        temporal = dp.build_temporal_sample(das, t, N)
        single   = dp.build_sample(das, t)

        if mh.is_unet_ready():
            prob_map = mh.predict_unet(temporal)
        else:
            label    = dp.make_label(single)
            prob_map = np.zeros((256, 256, 3), dtype=np.float32)
            for cls in range(3):
                prob_map[:,:,cls] = np.where(label == cls, 1.0, 0.0).astype(np.float32)

        pred_map = np.argmax(prob_map, axis=-1)
        conf_map = prob_map[:,:,2]

        land_mask = dp.get_land_mask()
        pred_map[land_mask] = 0
        conf_map[land_mask] = 0.0

        gps      = dp.pfz_to_gps(pred_map, conf_map, min_confidence=min_conf)

        df  = pd.DataFrame(gps)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)

        return send_file(
            buf,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'PFZ_GPS_t{t}.xlsx'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Training history ────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
def training_history():
    """Return training history for chart rendering."""
    try:
        history = mh.get_training_history()
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Cross-Validation Results ────────────────────────────────

@app.route('/api/cv_results', methods=['GET'])
def cv_results():
    """Return cross-validation results for both models."""
    try:
        results = {}
        models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')

        # UNET CV results
        unet_path = os.path.join(models_dir, 'cv_results_unet_temporal_block.json')
        if os.path.exists(unet_path):
            with open(unet_path) as f:
                results['unet'] = json.load(f)

        # ConvLSTM CV results
        lstm_path = os.path.join(models_dir, 'cv_results_convlstm_walk_forward.json')
        if os.path.exists(lstm_path):
            with open(lstm_path) as f:
                results['convlstm'] = json.load(f)

        return jsonify({'success': True, 'cv_results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cv_plot/<model>', methods=['GET'])
def cv_plot(model):
    """Serve CV plot PNG image."""
    models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    if model == 'unet':
        path = os.path.join(models_dir, 'cv_unet_temporal_block.png')
    elif model == 'convlstm':
        path = os.path.join(models_dir, 'cv_convlstm_walk_forward.png')
    else:
        return jsonify({'error': f'Unknown model: {model}'}), 404

    if not os.path.exists(path):
        return jsonify({'error': f'CV plot not found for {model}'}), 404

    return send_file(path, mimetype='image/png')


# ─── Upload NetCDF ───────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_nc():
    """Upload NetCDF files to server."""
    global _loaded, _das, _N_TIMES
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        uploaded = []
        for f in request.files.getlist('files'):
            if f.filename and f.filename.endswith('.nc'):
                filename = secure_filename(f.filename)
                path     = os.path.join(UPLOAD_DIR, filename)
                f.save(path)
                uploaded.append(filename)

        # Reload data
        _loaded = False
        _das    = None
        _N_TIMES = None

        return jsonify({
            'success':  True,
            'uploaded': uploaded,
            'count':    len(uploaded)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Map overlay data ────────────────────────────────────────

@app.route('/api/map_data', methods=['POST'])
def map_data():
    """Return PFZ grid as JSON for Leaflet overlay."""
    try:
        body = request.get_json(silent=True) or {}
        t    = int(body.get('time_index', 0))

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        t       = int(np.clip(t, 1, n_times - 2))
        temporal = dp.build_temporal_sample(das, t, N)
        single   = dp.build_sample(das, t)

        if mh.is_unet_ready():
            prob_map = mh.predict_unet(temporal)
        else:
            label    = dp.make_label(single)
            prob_map = np.zeros((256, 256, 3), dtype=np.float32)
            for cls in range(3):
                prob_map[:,:,cls] = np.where(label == cls, 1.0, 0.0).astype(np.float32)

        pred_map = np.argmax(prob_map, axis=-1)
        conf_map = prob_map[:,:,2]

        land_mask = dp.get_land_mask()
        pred_map[land_mask] = 0     # Force land to class 0
        conf_map[land_mask] = 0.0   # Force land confidence to 0

        # Downsample to 64x64 for JSON transfer
        from skimage.transform import resize as sk_resize
        pred_small = sk_resize(pred_map.astype(float), (64, 64),
                                order=0, anti_aliasing=False)
        conf_small = sk_resize(conf_map, (64, 64), anti_aliasing=True)

        LAT = np.linspace(dp.LAT_MIN, dp.LAT_MAX, 64).tolist()
        LON = np.linspace(dp.LON_MIN, dp.LON_MAX, 64).tolist()

        return jsonify({
            'success':     True,
            'pred_grid':   pred_small.astype(int).tolist(),
            'conf_grid':   conf_small.round(3).tolist(),
            'lat_range':   [dp.LAT_MIN, dp.LAT_MAX],
            'lon_range':   [dp.LON_MIN, dp.LON_MAX],
            'lat_ticks':   LAT,
            'lon_ticks':   LON,
            'time_index':  t,
            'date_str':    dp.get_date_for_index(das, t),
            'n_times':     n_times
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Tamil Nadu Coastal Fishing Locations ────────────────────

USER_LOCATIONS = [
    # NORTHERN COAST (Chennai & surroundings)
    {"name": "Chennai Harbour",        "lat": 13.0827, "lon": 80.2707},
    {"name": "Kasimedu",               "lat": 13.1200, "lon": 80.2900},
    {"name": "Ennore",                 "lat": 13.2167, "lon": 80.3167},
    {"name": "Pulicat",                "lat": 13.4167, "lon": 80.3167},
    {"name": "Pazhaverkadu",           "lat": 13.3833, "lon": 80.3333},

    # NORTH-CENTRAL COAST
    {"name": "Mahabalipuram",          "lat": 12.6269, "lon": 80.1927},
    {"name": "Kovalam (Chennai)",      "lat": 12.7897, "lon": 80.2519},
    {"name": "Sadras",                 "lat": 12.5500, "lon": 80.1833},
    {"name": "Kalpakkam",              "lat": 12.5500, "lon": 80.1667},
    {"name": "Marakkanam",             "lat": 12.2000, "lon": 79.9667},

    # CENTRAL COAST (Cuddalore & Villupuram)
    {"name": "Cuddalore",              "lat": 11.7480, "lon": 79.7680},
    {"name": "Parangipettai",          "lat": 11.4967, "lon": 79.7633},
    {"name": "Chidambaram Coast",      "lat": 11.3993, "lon": 79.6936},
    {"name": "Devanampattinam",        "lat": 11.5167, "lon": 79.8000},
    {"name": "Killai",                 "lat": 11.4667, "lon": 79.7833},

    # CAUVERY DELTA COAST (Nagapattinam)
    {"name": "Nagapattinam",           "lat": 10.7672, "lon": 79.8449},
    {"name": "Velankanni",             "lat": 10.6833, "lon": 79.8500},
    {"name": "Karaikal",               "lat": 10.9254, "lon": 79.8380},
    {"name": "Thondi",                 "lat": 10.7667, "lon": 79.6667},
    {"name": "Vedaranyam",             "lat": 10.3667, "lon": 79.8500},
    {"name": "Kodiyakarai",            "lat": 10.2833, "lon": 79.9000},

    # PALK BAY COAST (Thanjavur & Pudukkottai)
    {"name": "Pattukottai Coast",      "lat": 10.4167, "lon": 79.7833},
    {"name": "Adirampattinam",         "lat": 10.3500, "lon": 79.3833},
    {"name": "Mimisal",                "lat": 10.2167, "lon": 79.2500},
    {"name": "Muthupettai",            "lat": 10.4000, "lon": 79.5000},

    # RAMNAD COAST (Ramanathapuram)
    {"name": "Rameswaram",             "lat":  9.2880, "lon": 79.3129},
    {"name": "Mandapam",               "lat":  9.2833, "lon": 79.1333},
    {"name": "Pamban",                 "lat":  9.2833, "lon": 79.2167},
    {"name": "Kilakkarai",             "lat":  9.2333, "lon": 78.7833},
    {"name": "Ervadi",                 "lat":  9.1167, "lon": 78.7000},
    {"name": "Tuticorin (Thoothukudi)","lat":  8.7642, "lon": 78.1348},

    # SOUTHERN COAST (Tirunelveli & Kanniyakumari)
    {"name": "Colachel",               "lat":  8.1833, "lon": 77.2500},
    {"name": "Kanyakumari",            "lat":  8.0883, "lon": 77.5385},
    {"name": "Muttom",                 "lat":  8.1167, "lon": 77.3333},
    {"name": "Enayam",                 "lat":  8.2167, "lon": 77.2833},
    {"name": "Midalam",                "lat":  8.3000, "lon": 77.3167},
]


# ─── Haversine distance & compass bearing helpers ────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Compute Haversine distance in km between two (lat,lon) points."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compass_bearing(lat1, lon1, lat2, lon2):
    """Compute compass bearing from (lat1,lon1) to (lat2,lon2), return label."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    idx = int((bearing + 22.5) / 45) % 8
    return dirs[idx]


def latlon_to_grid(lat, lon):
    """Convert latitude/longitude to nearest grid pixel (i, j) in 256x256 grid."""
    # Clamp to grid bounds
    lat_c = max(dp.LAT_MIN, min(dp.LAT_MAX, lat))
    lon_c = max(dp.LON_MIN, min(dp.LON_MAX, lon))
    i = int(round((lat_c - dp.LAT_MIN) / (dp.LAT_MAX - dp.LAT_MIN) * 255))
    j = int(round((lon_c - dp.LON_MIN) / (dp.LON_MAX - dp.LON_MIN) * 255))
    return i, j


# ─── TASK 1: Location-wise Forecast ─────────────────────────

@app.route('/api/location_forecast', methods=['POST'])
def location_forecast():
    """
    Given user lat/lon (or a named location), extract the predicted PFZ class
    and confidence for each of the 7 forecast days at that exact pixel.
    Body JSON: { "lat": float, "lon": float } OR { "location_name": str }
    """
    try:
        body = request.get_json(silent=True) or {}

        # Resolve coordinates
        loc_name = body.get('location_name')
        if loc_name:
            loc = next((l for l in USER_LOCATIONS if l['name'] == loc_name), None)
            if not loc:
                return jsonify({'error': f'Unknown location: {loc_name}'}), 400
            lat, lon = loc['lat'], loc['lon']
        else:
            lat = float(body.get('lat', 0))
            lon = float(body.get('lon', 0))

        if lat == 0 and lon == 0:
            return jsonify({'error': 'Provide lat/lon or location_name'}), 400

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        i, j = latlon_to_grid(lat, lon)

        results = []
        for day in range(1, 8):
            pred_map, conf_map, land_mask, _ = _forecast_single_day(das, N, day, n_times)
            display_date = FORECAST_BASE_DATE + timedelta(days=day - 1)
            date_str = display_date.strftime('%Y-%m-%d')

            pfz_class = int(pred_map[i, j])
            confidence = float(conf_map[i, j])
            class_labels = {0: 'Low PFZ', 1: 'Medium PFZ', 2: 'High PFZ'}
            is_land = bool(land_mask[i, j])

            results.append({
                'day':          day,
                'date':         date_str,
                'pfz_class':    pfz_class,
                'pfz_label':    class_labels.get(pfz_class, 'Unknown'),
                'confidence':   round(confidence * 100, 2),
                'is_land':      is_land,
                'grid_i':       i,
                'grid_j':       j,
            })

        return jsonify({
            'success':       True,
            'location_name': loc_name or f'{lat:.4f}°N, {lon:.4f}°E',
            'lat':           lat,
            'lon':           lon,
            'grid_pixel':    {'i': i, 'j': j},
            'predictions':   results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── TASK 2: Most Probable PFZ Ranking ───────────────────────

@app.route('/api/pfz_ranking', methods=['POST'])
def pfz_ranking():
    """
    For each of the 7 days, scan the full 256×256 grid and find all
    High PFZ pixels (class=2) with confidence > 60%. Rank by confidence descending.
    Return top 10 coordinates per day.
    Body JSON: { "min_confidence": float (default 0.6), "top_n": int (default 10) }
    """
    try:
        body = request.get_json(silent=True) or {}
        min_conf = float(body.get('min_confidence', 0.6))
        top_n = int(body.get('top_n', 10))

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        LAT = np.linspace(dp.LAT_MIN, dp.LAT_MAX, 256)
        LON = np.linspace(dp.LON_MIN, dp.LON_MAX, 256)

        all_days = []
        for day in range(1, 8):
            pred_map, conf_map, land_mask, _ = _forecast_single_day(das, N, day, n_times)
            display_date = FORECAST_BASE_DATE + timedelta(days=day - 1)
            date_str = display_date.strftime('%Y-%m-%d')

            # Find all High PFZ pixels with confidence > threshold
            rows, cols = np.where(
                (pred_map == 2) & (conf_map >= min_conf) & (~land_mask)
            )

            candidates = []
            for r, c in zip(rows, cols):
                candidates.append({
                    'lat':        float(np.round(LAT[r], 4)),
                    'lon':        float(np.round(LON[c], 4)),
                    'confidence': float(np.round(conf_map[r, c] * 100, 2)),
                    'pfz_class':  2,
                    'pfz_label':  'High PFZ',
                })

            # Sort by confidence descending, take top N
            candidates.sort(key=lambda x: -x['confidence'])
            top_candidates = candidates[:top_n]

            all_days.append({
                'day':         day,
                'date':        date_str,
                'total_high':  len(candidates),
                'top_zones':   top_candidates
            })

        return jsonify({
            'success':         True,
            'min_confidence':  min_conf * 100,
            'top_n':           top_n,
            'rankings':        all_days
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── TASK 3 & 4: CSV Export with Distance & Bearing ──────────

@app.route('/api/pfz_csv', methods=['GET', 'POST'])
def pfz_csv():
    """
    Export top ranked High PFZ coordinates for all 7 days as CSV.
    Includes Haversine distance and compass bearing from user location.
    GET params or POST body:
      lat, lon — user's location (required)
      min_confidence — float (default 0.6)
      top_n — int (default 10)
      location_name — optional, for header
    """
    try:
        if request.method == 'GET':
            lat = float(request.args.get('lat', 0))
            lon = float(request.args.get('lon', 0))
            min_conf = float(request.args.get('min_confidence', 0.6))
            top_n = int(request.args.get('top_n', 10))
            loc_name = request.args.get('location_name', '')
        else:
            body = request.get_json(silent=True) or {}
            lat = float(body.get('lat', 0))
            lon = float(body.get('lon', 0))
            min_conf = float(body.get('min_confidence', 0.6))
            top_n = int(body.get('top_n', 10))
            loc_name = body.get('location_name', '')

        # Resolve named location
        if loc_name and (lat == 0 and lon == 0):
            loc = next((l for l in USER_LOCATIONS if l['name'] == loc_name), None)
            if loc:
                lat, lon = loc['lat'], loc['lon']

        if lat == 0 and lon == 0:
            return jsonify({'error': 'Provide lat/lon or location_name'}), 400

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        LAT = np.linspace(dp.LAT_MIN, dp.LAT_MAX, 256)
        LON = np.linspace(dp.LON_MIN, dp.LON_MAX, 256)

        rows_out = []
        for day in range(1, 8):
            pred_map, conf_map, land_mask, _ = _forecast_single_day(das, N, day, n_times)
            display_date = FORECAST_BASE_DATE + timedelta(days=day - 1)
            date_str = display_date.strftime('%Y-%m-%d')

            # Find High PFZ pixels
            rr, cc = np.where(
                (pred_map == 2) & (conf_map >= min_conf) & (~land_mask)
            )

            candidates = []
            for r, c in zip(rr, cc):
                pt_lat = float(np.round(LAT[r], 4))
                pt_lon = float(np.round(LON[c], 4))
                conf_pct = float(np.round(conf_map[r, c] * 100, 2))
                dist = round(haversine_km(lat, lon, pt_lat, pt_lon), 2)
                bearing = compass_bearing(lat, lon, pt_lat, pt_lon)
                candidates.append({
                    'Day':            day,
                    'Date':           date_str,
                    'Latitude':       pt_lat,
                    'Longitude':      pt_lon,
                    'PFZ_Class':      'High PFZ',
                    'Confidence_%':   conf_pct,
                    'Distance_km':    dist,
                    'Bearing':        bearing,
                })

            candidates.sort(key=lambda x: (x['Distance_km'], -x['Confidence_%']))
            rows_out.extend(candidates)

        # Build CSV in-memory
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=[
            'Day', 'Date', 'Latitude', 'Longitude',
            'PFZ_Class', 'Confidence_%', 'Distance_km', 'Bearing'
        ])
        writer.writeheader()
        writer.writerows(rows_out)

        output = io.BytesIO()
        output.write(buf.getvalue().encode('utf-8'))
        output.seek(0)

        safe_name = loc_name.replace(' ', '_') if loc_name else f'{lat}_{lon}'
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'PFZ_Top_Zones_{safe_name}.csv'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── User locations list endpoint ────────────────────────────

@app.route('/api/user_locations', methods=['GET'])
def user_locations():
    """Return the list of predefined Tamil Nadu fishing locations."""
    return jsonify({'success': True, 'locations': USER_LOCATIONS})


# ─── TASK 2+4 JSON with distance (for UI table rendering) ───

@app.route('/api/pfz_ranking_with_distance', methods=['POST'])
def pfz_ranking_with_distance():
    """
    Combined Task 2 + Task 4: PFZ ranking with distance from user.
    Body JSON: { "lat": float, "lon": float, "min_confidence": 0.6, "top_n": 10, "location_name": "" }
    Returns JSON with rankings including distance_km and bearing per coordinate.
    """
    try:
        body = request.get_json(silent=True) or {}
        lat = float(body.get('lat', 0))
        lon = float(body.get('lon', 0))
        loc_name = body.get('location_name', '')
        min_conf = float(body.get('min_confidence', 0.6))
        top_n = int(body.get('top_n', 10))

        if loc_name and (lat == 0 and lon == 0):
            loc = next((l for l in USER_LOCATIONS if l['name'] == loc_name), None)
            if loc:
                lat, lon = loc['lat'], loc['lon']

        if lat == 0 and lon == 0:
            return jsonify({'error': 'Provide lat/lon or location_name'}), 400

        das, N = get_data()
        if das is None or N is None:
            return jsonify({'error': 'Data not loaded'}), 400

        n_times = int(N) if N is not None else 0
        LAT = np.linspace(dp.LAT_MIN, dp.LAT_MAX, 256)
        LON = np.linspace(dp.LON_MIN, dp.LON_MAX, 256)

        all_days = []
        for day in range(1, 8):
            pred_map, conf_map, land_mask, _ = _forecast_single_day(das, N, day, n_times)
            display_date = FORECAST_BASE_DATE + timedelta(days=day - 1)
            date_str = display_date.strftime('%Y-%m-%d')

            rr, cc = np.where(
                (pred_map == 2) & (conf_map >= min_conf) & (~land_mask)
            )

            candidates = []
            for r, c in zip(rr, cc):
                pt_lat = float(np.round(LAT[r], 4))
                pt_lon = float(np.round(LON[c], 4))
                conf_pct = float(np.round(conf_map[r, c] * 100, 2))
                dist = round(haversine_km(lat, lon, pt_lat, pt_lon), 2)
                bearing = compass_bearing(lat, lon, pt_lat, pt_lon)
                candidates.append({
                    'lat':          pt_lat,
                    'lon':          pt_lon,
                    'confidence':   conf_pct,
                    'distance_km':  dist,
                    'bearing':      bearing,
                })

            candidates.sort(key=lambda x: (x['distance_km'], -x['confidence']))
            all_days.append({
                'day':         day,
                'date':        date_str,
                'total_high':  len(candidates),
                'top_zones':   candidates[:top_n]
            })

        return jsonify({
            'success':         True,
            'location_name':   loc_name or f'{lat:.4f}°N, {lon:.4f}°E',
            'user_lat':        lat,
            'user_lon':        lon,
            'min_confidence':  min_conf * 100,
            'top_n':           top_n,
            'rankings':        all_days
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Serve frontend ──────────────────────────────────────────

@app.route('/')
def serve_frontend():
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))


# ─── Main ────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--port',     type=int, default=5001)
    parser.add_argument('--debug',    action='store_true')
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = args.data_dir
        os.environ['PFZ_DATA_DIR'] = args.data_dir

    print("\n🎣 PFZ Backend API Server")
    print(f"   Data directory : {DATA_DIR}")
    print(f"   Port           : {args.port}")
    print(f"   Upload dir     : {UPLOAD_DIR}")
    print(f"   Frontend dir   : {FRONTEND_DIR}")
    print(f"   Open browser   : http://localhost:{args.port}")

    # Pre-load models on startup
    mh.load_models()

    # Pre-load data if available
    get_data()

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
