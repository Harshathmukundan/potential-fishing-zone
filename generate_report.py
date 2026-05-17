import json
import base64
import os

def b64_img(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

unet_cv = json.load(open("backend/saved_models/cv_results_unet_temporal_block.json"))
lstm_cv = json.load(open("backend/saved_models/cv_results_convlstm_walk_forward.json"))

unet_img = b64_img("backend/saved_models/cv_unet_temporal_block.png")
lstm_img = b64_img("backend/saved_models/cv_convlstm_walk_forward.png")

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PFZ Navigator - Project Report</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; padding: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 40px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
        h3 {{ color: #16a085; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #f8f9fa; color: #333; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric-box {{ background: #f1f8ff; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ccc; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .footer {{ margin-top: 50px; font-size: 0.9em; color: #7f8c8d; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }}
    </style>
</head>
<body>

<h1>PFZ Navigator — AI-Based Geospatial Decision Support System</h1>
<p><strong>Authors:</strong> Harshath Mukundan & Praveen N<br>
<strong>Institution:</strong> SRMIST</p>

<div class="metric-box">
    <strong>Abstract:</strong> This project presents a full end-to-end deep learning system for identifying and forecasting Potential Fishing Zones (PFZ) in the Bay of Bengal. It utilizes multi-channel satellite oceanography data processed through U-Net (for spatial segmentation) and ConvLSTM (for spatio-temporal forecasting). The system includes a fully functional dashboard for interactive geographic visualization and cross-validated prediction analysis.
</div>

<h2>1. System Architecture & Setup</h2>
<p>The PFZ Navigator acts as a decision support pipeline. Raw NetCDF satellite data is uploaded via the web interface. The backend processes the following parameters: <strong>Sea Surface Temperature (SST), Sea Surface Height (SSH), U-Currents, V-Currents, and Chlorophyll-a</strong>. These are regridded to a 256×256 standardized grid covering the Tamil Nadu coast.</p>

<p>The system is split into:</p>
<ul>
    <li><strong>Model Backend Pipeline:</strong> Data loaders, training scripts, and rigorous cross-validation runners.</li>
    <li><strong>Flask REST API:</strong> Exposes endpoints to run predictions, forecasts, and convert PFZ pixels to exportable GPS coordinates.</li>
    <li><strong>Interactive Dashboard:</strong> A frontend written in vanilla JS/CSS/HTML incorporating Leaflet mapping and Chart.js analytics for intuitive user interaction.</li>
</ul>

<h2>2. Artificial Intelligence Models</h2>

<h3>U-Net (Spatial Evaluator)</h3>
<p>The U-Net model assesses the current ocean state to map instantaneous PFZs. It utilizes an encoder-decoder architecture with skip connections to preserve high-resolution spatial boundaries of thermal fronts and upwellings.</p>
<ul>
    <li><strong>Input Shape:</strong> (256, 256, 24) — 8 channels over 3 preceding days.</li>
    <li><strong>Output Shape:</strong> (256, 256, 3) — Pixel-wise classification (Low, Medium, High).</li>
</ul>

<h3>ConvLSTM (Spatio-Temporal Forecaster)</h3>
<p>The ConvLSTM network models both the spatial context and the temporal evolution of ocean parameters. This allows for an unprecedented 7-day forward prediction of dynamic ocean fronts.</p>
<ul>
    <li><strong>Input Shape:</strong> (7, 256, 256, 8) — A sequence of the past 7 daily states.</li>
    <li><strong>Output Shape:</strong> (256, 256, 3) — The future day's classification map.</li>
</ul>

<h2>3. Cross-Validation Methodology</h2>
<p>To ensure robust results that translate well to operational real-world scenarios, sophisticated cross-validation boundaries were enforced:</p>
<ul>
    <li><strong>Temporal Block Split (U-Net, 3-fold):</strong> The dataset timeline was strictly divided into non-overlapping temporal blocks. This guarantees no leakage of proximate, highly-correlated days between training and validation sets.</li>
    <li><strong>Walk-Forward Split (ConvLSTM, 3-fold):</strong> The time-series nature of forecasting requires an expanding window. The model trains strictly on contiguous past data to evaluate its performance exclusively on unseen future sequences.</li>
</ul>

<h2>4. Evaluation Metrics & Results</h2>

<h3>4.1 U-Net (Temporal Block CV Results)</h3>
<table>
    <tr><th>Metric</th><th>Mean (3-fold)</th><th>95% CI</th></tr>
    <tr><td>Accuracy</td><td>{unet_cv['accuracy']['mean']:.4f}</td><td>±{unet_cv['accuracy']['ci95']:.4f}</td></tr>
    <tr><td>Macro Precision</td><td>{unet_cv['precision_macro']['mean']:.4f}</td><td>±{unet_cv['precision_macro']['ci95']:.4f}</td></tr>
    <tr><td>Macro Recall</td><td>{unet_cv['recall_macro']['mean']:.4f}</td><td>±{unet_cv['recall_macro']['ci95']:.4f}</td></tr>
    <tr><td>Macro F1-Score</td><td>{unet_cv['f1_macro']['mean']:.4f}</td><td>±{unet_cv['f1_macro']['ci95']:.4f}</td></tr>
    <tr><td>Cohen's Kappa</td><td>{unet_cv['kappa']['mean']:.4f}</td><td>±{unet_cv['kappa']['ci95']:.4f}</td></tr>
    <tr><td>mIoU</td><td>{unet_cv['miou']['mean']:.4f}</td><td>±{unet_cv['miou']['ci95']:.4f}</td></tr>
</table>
<p><em>Figure 1: U-Net Cross-Validation Visualizations (Accuracy, Loss, Confusion Matrix, and Radar Chart)</em></p>
<img src="{unet_img}" alt="U-Net Metrics Plot">

<h3>4.2 ConvLSTM (Walk-Forward CV Results)</h3>
<table>
    <tr><th>Metric</th><th>Mean (3-fold)</th><th>95% CI</th></tr>
    <tr><td>Accuracy</td><td>{lstm_cv['accuracy']['mean']:.4f}</td><td>±{lstm_cv['accuracy']['ci95']:.4f}</td></tr>
    <tr><td>Macro Precision</td><td>{lstm_cv['precision_macro']['mean']:.4f}</td><td>±{lstm_cv['precision_macro']['ci95']:.4f}</td></tr>
    <tr><td>Macro Recall</td><td>{lstm_cv['recall_macro']['mean']:.4f}</td><td>±{lstm_cv['recall_macro']['ci95']:.4f}</td></tr>
    <tr><td>Macro F1-Score</td><td>{lstm_cv['f1_macro']['mean']:.4f}</td><td>±{lstm_cv['f1_macro']['ci95']:.4f}</td></tr>
    <tr><td>Cohen's Kappa</td><td>{lstm_cv['kappa']['mean']:.4f}</td><td>±{lstm_cv['kappa']['ci95']:.4f}</td></tr>
    <tr><td>mIoU</td><td>{lstm_cv['miou']['mean']:.4f}</td><td>±{lstm_cv['miou']['ci95']:.4f}</td></tr>
</table>
<p><em>Figure 2: ConvLSTM Cross-Validation Visualizations (Accuracy, Loss, Confusion Matrix, and Radar Chart)</em></p>
<img src="{lstm_img}" alt="ConvLSTM Metrics Plot">

<h2>5. Conclusion</h2>
<p>The <strong>PFZ Navigator</strong> proves the efficacy of applying advanced deep learning to oceanic satellite data. The U-Net model demonstrates high precision in outlining front boundaries, while the ConvLSTM leverages temporal dynamics to offer functional predictive insights. The integrated web utility empowers fisheries and environmental agencies to make actionable, GPS-guided decisions driven by data.</p>

<div class="footer">
    Generated via automated report pipeline • PFZ Navigator Project
</div>

</body>
</html>
"""

with open("PFZ_Project_Report.html", "w") as f:
    f.write(html)

print("Report generated successfully: PFZ_Project_Report.html")
