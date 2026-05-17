# 🎣 PFZ Navigator — AI-Based Geospatial Decision Support System

> **End-to-End Potential Fishing Zone (PFZ) Identification System** utilizing deep learning (U-Net and ConvLSTM) on CMEMS satellite oceanographic data for the Tamil Nadu Coast and Bay of Bengal.

**SRMIST — Harshath Mukundan & Praveen N**

---

## 🌊 1. Project Overview & Objective

The **PFZ Navigator** is a comprehensive, AI-powered geospatial decision support system designed to identify and forecast Potential Fishing Zones (PFZs). Traditional PFZ advisories rely on rudimentary thresholding of sea surface temperatures and chlorophyll levels, which often miss complex, non-linear spatiotemporal ocean dynamics. 

This project solves that limitation by employing advanced Deep Learning techniques—specifically **U-Net** for spatial segmentation and **ConvLSTM** for spatiotemporal forecasting—to predict PFZs with high accuracy. The end result is a full-stack web application that allows researchers and fisheries to upload raw satellite data, run AI predictions, view real-time maps, and export high-confidence GPS coordinates.

### Study Region
- **Latitude**: 5.0°N — 14.5°N
- **Longitude**: 77.0°E — 84.92°E
- **Coverage**: Bay of Bengal / Tamil Nadu eastern coast

---

## 🛰️ 2. Data Pipeline & Inputs

The system ingests raw NetCDF (`.nc`) files from the **Copernicus Marine Environment Monitoring Service (CMEMS)**. It processes five critical oceanographic parameters to deduce upwelling zones and thermal fronts where fish naturally aggregate:

| Parameter | Source Dataset | Physical Meaning | Role in PFZ |
|-----------|----------------|------------------|-------------|
| **SST** | CMEMS SST (thetao) | Sea Surface Temperature | Fish seek specific temperature gradients (fronts). |
| **SSH** | CMEMS ZOS | Sea Surface Height | Identifies oceanic eddies and upwelling/downwelling. |
| **U-Current** | CMEMS UO | Eastward water velocity | Indicates water mass movement and mixing. |
| **V-Current** | CMEMS VO | Northward water velocity| Indicates water mass movement and mixing. |
| **Chlorophyll-a** | CMEMS CHL | Plankton biomass | Primary indicator of food availability for fish. |

**Preprocessing Pipeline:**
1. **Regridding**: Aligns all variables to a uniform `256×256` spatial grid using bounding box coordinates.
2. **Normalization**: Applies Min-Max scaling to each channel across the historical timeline.
3. **Land Masking**: Applies a static binary mask to exclude landmasses from predictions.

---

## 🧠 3. Artificial Intelligence Models

The core of the system relies on two distinct deep learning architectures optimized for geospatial data.

### Model 1: U-Net (Spatial Classification)
- **Objective**: Identify PFZs in real-time based on the current day's oceanographic conditions.
- **Input**: A `256×256×24` tensor (8 channels over 3 sequential time steps to capture immediate variance).
- **Architecture**: A fully convolutional encoder-decoder network with skip connections that preserve high-resolution spatial details.
- **Output**: A `256×256×3` categorical probability map classifying each pixel as **Low**, **Medium**, or **High** PFZ.

### Model 2: ConvLSTM (Spatio-Temporal Forecasting)
- **Objective**: Forecast future PFZs up to 7 days in advance.
- **Input**: A sequence of 7 daily frames, each `256×256×8` channels.
- **Architecture**: Convolutional Long Short-Term Memory (ConvLSTM2D) layers that learn both the spatial layout of the ocean and the temporal evolution of currents and temperatures over time.
- **Output**: A 7-day ahead `256×256×3` PFZ probability forecast.

---

## 🔬 4. Cross-Validation & Methodology

To ensure research-grade validity and prevent data leakage, the models are evaluated using rigorous cross-validation methodologies:

- **Temporal Block Split (for U-Net)**: The dataset is split into defined, contiguous blocks of time. This prevents the model from interpolating between temporally adjacent (and highly correlated) frames.
- **Walk-Forward Split (for ConvLSTM)**: Mimicking real-world forecasting, the model is trained on a strictly historical window to predict the immediate future, stepping forward sequentially.
- **Metrics Tracked**: Pixel-wise Accuracy, Macro F1-Score, Cohen's Kappa (to account for class imbalance), and mean Intersection over Union (mIoU).

---

## 🏗️ 5. System Architecture

The project is structured as a modern standard full-stack application:

```text
pfz_project/
├── model/                    # ML Training & Evaluation Pipeline
│   ├── train.py              # Main training script (trains U-Net & ConvLSTM)
│   ├── cross_validate.py     # Rigorous k-fold evaluation & metric generation
│   ├── dataset.py            # Custom tf.keras.utils.Sequence data loaders
│   ├── unet.py               # U-Net Topologies
│   └── convlstm.py           # ConvLSTM Topologies
│
├── backend/                  # Flask REST API Server 
│   ├── app.py                # Main backend exposing 8+ API endpoints
│   ├── data_processor.py     # NetCDF IO, regridding, and interpolation
│   ├── model_handler.py      # .keras model loading and inference logic
│   ├── saved_models/         # Compiled models and CV JSON/PNG reports
│   └── uploads/DATA/         # Directory for CMEMS NetCDF datasets
│
└── frontend/                 # Interactive Dashboard (Vanilla JS/CSS/HTML)
    ├── index.html            # User interface structure
    ├── styles/main.css       # Dark ocean-tech aesthetic styling
    └── src/
        ├── api.js            # Network layer (fetch with timeouts)
        ├── app.js            # UI state, event listeners, logic
        ├── charts.js         # Chart.js integration (Metrics, Radar, Donuts)
        └── map.js            # Leaflet.js interactive maps and heatmaps
```

---

## 🖥️ 6. Dashboard Features

The web frontend (`frontend/index.html`) offers 6 rich, interactive panels:

1. **📊 Dashboard**: Displays real-time system status, overall model accuracies, training loss curves, and immediate PFZ distributions.
2. **⬡ Predict (U-Net)**: Allows the user to select any historical time index, run the U-Net model on-demand, visualize the PFZ heatmap, and export high-confidence target coordinates to a CSV.
3. **◷ Forecast (ConvLSTM)**: Offers 7-day forward predictions. Users click a day (e.g., "Day 3"), view the predicted ocean state map, and download the forecasted GPS targets.
4. **◉ Live Map**: An interactive Leaflet map that overlays the generated AI predictions geographically over the coast of Tamil Nadu.
5. **🔬 Research**: A dedicated academic panel that juxtaposes U-Net and ConvLSTM performance. It displays the cross-validation methodology, per-class metrics tables, side-by-side metric comparisons, and the publication-ready 6-panel evaluation plots.
6. **⊕ Upload Data**: A drag-and-drop interface allowing administrators to upload new CMEMS NetCDF files directly into the backend processing engine.

---

## 🚀 7. Setup & Execution Instructions

### Prerequisites
- Python 3.10+
- ~2GB of disk space for models and environmental NetCDF data.

### Step 1: Install Dependencies
```bash
cd pfz_project
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### Step 2: Data Placement
Place the required 5 CMEMS `.nc` files in `backend/uploads/DATA/`.

### Step 3: Train & Validate Models (Optional - Models are pre-saved)
```bash
cd model
source ../venv/bin/activate

# 1. Train models
python train.py --data_dir ../backend/uploads/DATA

# 2. Generate evaluation reports & plots
python cross_validate.py --data_dir ../backend/uploads/DATA
```

### Step 4: Start the Backend API
```bash
cd backend
source ../venv/bin/activate
python app.py --port 5001
```
*The backend will load the `.keras` models into RAM and expose the `http://localhost:5001/api/` endpoints.*

### Step 5: Launch the Frontend UI
Simply open `frontend/index.html` in any modern web browser (Chrome/Safari/Firefox). The frontend will automatically connect to the local Flask backend, verify system health, and unlock the interactive dashboard.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
