"""
cross_validate.py — Research-Grade Cross-Validation for PFZ Models
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N

Implements three CV strategies suitable for spatiotemporal oceanographic data:
  1. Temporal Block CV   — non-overlapping temporal blocks (prevents data leakage)
  2. Walk-Forward CV     — train on past, validate on future (mimics deployment)
  3. Stratified K-Fold   — balanced class distribution per fold

Usage:
    python cross_validate.py --data_dir ../backend/uploads/DATA --method temporal_block
    python cross_validate.py --data_dir ../backend/uploads/DATA --method walk_forward
    python cross_validate.py --data_dir ../backend/uploads/DATA --method stratified
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import mixed_precision  # type: ignore
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    cohen_kappa_score
)
from scipy import stats as scipy_stats

from unet     import build_unet
from convlstm import build_convlstm
from dataset  import load_datasets, PFZGenerator, ConvLSTMGenerator

# ─── Config ──────────────────────────────────────────────────
SEED   = 42
TARGET = 256
np.random.seed(SEED)
tf.random.set_seed(SEED)
mixed_precision.set_global_policy('mixed_float16')

CLASS_NAMES = ['Low PFZ', 'Medium PFZ', 'High PFZ']


# ═══════════════════════════════════════════════════════════════
# CV SPLIT STRATEGIES
# ═══════════════════════════════════════════════════════════════

def temporal_block_splits(N_TIMES, k=5):
    """
    Temporal Block Cross-Validation (recommended for spatiotemporal data).

    Divides the time axis into k non-overlapping contiguous blocks.
    Each fold uses one block as validation and the remaining as training.
    Preserves temporal structure and prevents data leakage.

    Reference: Roberts et al. (2017) "Cross-validation strategies for data
    with temporal, spatial, hierarchical, or phylogenetic structure"
    """
    indices = np.arange(1, N_TIMES - 1)
    block_size = len(indices) // k
    folds = []

    for i in range(k):
        val_start = i * block_size
        val_end = val_start + block_size if i < k - 1 else len(indices)
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        folds.append((train_idx, val_idx))
        print(f"  Fold {i+1}: Train={len(train_idx)} samples "
              f"(t={indices[0]}..{indices[val_start-1] if val_start > 0 else 'N/A'}, "
              f"t={indices[val_end] if val_end < len(indices) else 'N/A'}..{indices[-1]}), "
              f"Val={len(val_idx)} samples (t={val_idx[0]}..{val_idx[-1]})")

    return folds


def walk_forward_splits(N_TIMES, k=5, min_train_ratio=0.3):
    """
    Walk-Forward (Expanding Window) Cross-Validation.

    Train on all data up to time t, validate on the next block.
    Mimics real-world deployment: always train on past, test on future.
    Most realistic for forecasting evaluation.

    Reference: Tashman (2000) "Out-of-sample tests of forecasting accuracy"
    """
    indices = np.arange(1, N_TIMES - 1)
    n = len(indices)
    min_train = int(n * min_train_ratio)
    step = (n - min_train) // k
    folds = []

    for i in range(k):
        train_end = min_train + i * step
        val_end = min(train_end + step, n)
        if train_end >= n or val_end > n:
            break
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        folds.append((train_idx, val_idx))
        print(f"  Fold {i+1}: Train={len(train_idx)} samples "
              f"(t={train_idx[0]}..{train_idx[-1]}), "
              f"Val={len(val_idx)} samples (t={val_idx[0]}..{val_idx[-1]})")

    return folds


def stratified_kfold_splits(das, N_TIMES, k=5):
    """
    Stratified K-Fold Cross-Validation.

    Ensures each fold has approximately the same class distribution.
    Uses the dominant PFZ class per time step for stratification.

    Note: This does NOT preserve temporal ordering — use only as a
    secondary validation alongside temporal methods.
    """
    from dataset import build_sample, make_label

    indices = np.arange(1, N_TIMES - 1)

    # Determine dominant class per time step (subsample for speed)
    print("  Computing class distribution per time step...")
    labels = []
    for t in indices[::max(1, len(indices) // 500)]:  # sample up to 500
        sample = build_sample(das, int(t))
        label = make_label(sample)
        dominant = int(np.argmax(np.bincount(label.flatten(), minlength=3)))
        labels.append(dominant)

    # Interpolate for all indices
    labels_full = np.interp(
        np.arange(len(indices)),
        np.linspace(0, len(indices) - 1, len(labels)),
        labels
    ).astype(int)

    # Stratified split
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = []
    for i, (train_i, val_i) in enumerate(skf.split(indices, labels_full)):
        train_idx = indices[train_i]
        val_idx = indices[val_i]
        folds.append((train_idx, val_idx))
        train_dist = np.bincount(labels_full[train_i], minlength=3)
        val_dist = np.bincount(labels_full[val_i], minlength=3)
        print(f"  Fold {i+1}: Train={len(train_idx)} "
              f"(L:{train_dist[0]} M:{train_dist[1]} H:{train_dist[2]}), "
              f"Val={len(val_idx)} "
              f"(L:{val_dist[0]} M:{val_dist[1]} H:{val_dist[2]})")

    return folds


# ═══════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════

def evaluate_fold(model, val_gen, fold_num):
    """
    Comprehensive per-fold evaluation metrics for research paper.

    Returns dict with: accuracy, precision, recall, F1, IoU, kappa,
    confusion matrix, and per-class metrics.
    """
    all_preds, all_trues = [], []

    for i in range(len(val_gen)):
        X, Y = val_gen[i]
        pred = model.predict(X, verbose=0)
        all_preds.append(np.argmax(pred, axis=-1).flatten())
        all_trues.append(np.argmax(Y, axis=-1).flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    # IoU per class (Jaccard Index)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    iou_per = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-8)
    miou = np.mean(iou_per)

    metrics = {
        'fold': fold_num,
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'kappa': float(kappa),
        'miou': float(miou),
        'per_class': {
            cls: {
                'precision': float(prec_per[i]),
                'recall': float(rec_per[i]),
                'f1': float(f1_per[i]),
                'iou': float(iou_per[i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        },
        'confusion_matrix': cm.tolist(),
    }

    print(f"\n  ── Fold {fold_num} Results ──")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 (macro)   : {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Cohen's κ    : {kappa:.4f}")
    print(f"  mIoU         : {miou:.4f}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:12s}  P={prec_per[i]:.3f}  R={rec_per[i]:.3f}  "
              f"F1={f1_per[i]:.3f}  IoU={iou_per[i]:.3f}")

    return metrics


# ═══════════════════════════════════════════════════════════════
# CROSS-VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════

def run_cv(data_dir, method='temporal_block', k=5, epochs=5,
           batch_size=4, model_type='unet', output_dir='../backend/saved_models'):
    """Run k-fold cross-validation and generate publication-ready results."""

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION: {method.upper()} ({k}-fold)")
    print(f"  Model: {model_type.upper()} | Epochs: {epochs}")
    print(f"{'='*60}")

    print("\n[1] Loading datasets...")
    das, N_TIMES = load_datasets(data_dir)

    # Generate folds
    print(f"\n[2] Generating {method} splits...")
    if method == 'temporal_block':
        folds = temporal_block_splits(N_TIMES, k)
    elif method == 'walk_forward':
        folds = walk_forward_splits(N_TIMES, k)
    elif method == 'stratified':
        folds = stratified_kfold_splits(das, N_TIMES, k)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Run each fold
    all_metrics = []
    all_histories = []

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_i + 1
        print(f"\n{'─'*60}")
        print(f"  FOLD {fold_num}/{k}")
        print(f"{'─'*60}")

        # Build generators
        if model_type == 'unet':
            train_gen = PFZGenerator(das, train_idx, N_TIMES, batch_size=batch_size)
            val_gen = PFZGenerator(das, val_idx, N_TIMES, batch_size=batch_size, shuffle=False)
            model = build_unet(input_channels=24)
        else:
            train_gen = ConvLSTMGenerator(das, train_idx, N_TIMES, batch_size=batch_size)
            val_gen = ConvLSTMGenerator(das, val_idx, N_TIMES, batch_size=batch_size)
            model = build_convlstm()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),  # type: ignore
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(  # type: ignore
                monitor='val_accuracy', patience=5,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(  # type: ignore
                monitor='val_loss', patience=3,
                factor=0.5, min_lr=1e-6, verbose=1
            ),
        ]

        # Train
        history = model.fit(
            train_gen, validation_data=val_gen,
            epochs=epochs, callbacks=callbacks, verbose=1
        )

        all_histories.append({
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
        })

        # Evaluate
        metrics = evaluate_fold(model, val_gen, fold_num)
        all_metrics.append(metrics)

        # Clear GPU memory
        tf.keras.backend.clear_session()

    # ── Aggregate results ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AGGREGATED RESULTS ({method}, {k}-fold)")
    print(f"{'='*60}")

    summary = aggregate_results(all_metrics, method, k)
    summary['model_type'] = model_type
    summary['method'] = method
    summary['k'] = k
    summary['epochs'] = epochs
    summary['fold_metrics'] = all_metrics
    summary['fold_histories'] = all_histories

    # Save results
    out_path = os.path.join(output_dir, f'cv_results_{model_type}_{method}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # Generate plots
    plot_cv_results(all_metrics, all_histories, model_type, method, output_dir)

    return summary


def aggregate_results(all_metrics, method, k):
    """
    Compute mean ± std across folds for each metric.
    For research papers: report mean ± std with 95% CI.
    """
    metric_keys = ['accuracy', 'precision_macro', 'recall_macro',
                   'f1_macro', 'f1_weighted', 'kappa', 'miou']

    summary = {}
    values = {}

    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        values[key] = vals
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)  # sample std
        ci95 = 1.96 * std / np.sqrt(len(vals))  # 95% confidence interval
        summary[key] = {
            'mean': float(mean),
            'std': float(std),
            'ci95': float(ci95),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'per_fold': [float(v) for v in vals],
        }
        print(f"  {key:18s}: {mean:.4f} ± {std:.4f}  (95% CI: ±{ci95:.4f})")

    # Per-class aggregation
    summary['per_class'] = {}
    for cls in CLASS_NAMES:
        summary['per_class'][cls] = {}
        for sub_key in ['precision', 'recall', 'f1', 'iou']:
            vals = [m['per_class'][cls][sub_key] for m in all_metrics]
            summary['per_class'][cls][sub_key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
            }

    print(f"\n  Per-class metrics (mean ± std):")
    print(f"  {'Class':14s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s} {'IoU':>12s}")
    print(f"  {'─'*62}")
    for cls in CLASS_NAMES:
        pc = summary['per_class'][cls]
        print(f"  {cls:14s} "
              f"{pc['precision']['mean']:.3f}±{pc['precision']['std']:.3f}  "
              f"{pc['recall']['mean']:.3f}±{pc['recall']['std']:.3f}  "
              f"{pc['f1']['mean']:.3f}±{pc['f1']['std']:.3f}  "
              f"{pc['iou']['mean']:.3f}±{pc['iou']['std']:.3f}")

    # Aggregate confusion matrix
    agg_cm = np.sum([np.array(m['confusion_matrix']) for m in all_metrics], axis=0)
    summary['aggregated_confusion_matrix'] = agg_cm.tolist()

    return summary


# ═══════════════════════════════════════════════════════════════
# PUBLICATION-READY PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_cv_results(all_metrics, all_histories, model_type, method, output_dir):
    """Generate publication-quality figures."""

    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.suptitle(f'{model_type.upper()} — {method.replace("_", " ").title()} '
                 f'Cross-Validation Results', fontsize=16, fontweight='bold', y=0.98)

    # 1) Per-fold accuracy bar chart
    ax1 = fig.add_subplot(2, 3, 1)
    folds = [m['fold'] for m in all_metrics]
    accs = [m['accuracy'] for m in all_metrics]
    colors = ['#3498db'] * len(folds)
    bars = ax1.bar(folds, accs, color=colors, edgecolor='#2176ae', linewidth=0.8)
    mean_acc = np.mean(accs)
    ax1.axhline(mean_acc, color='#e74c3c', linestyle='--', linewidth=1.5,
                label=f'Mean = {mean_acc:.4f}')
    ax1.set_xlabel('Fold', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Per-Fold Accuracy', fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim([min(accs) - 0.05, 1.0])
    ax1.grid(axis='y', alpha=0.3)

    # 2) Learning curves (all folds overlaid)
    ax2 = fig.add_subplot(2, 3, 2)
    for i, h in enumerate(all_histories):
        alpha = 0.4
        ax2.plot(h['val_accuracy'], alpha=alpha, color='#3498db', linewidth=1)
        ax2.plot(h['accuracy'], alpha=alpha, color='#f5a623', linewidth=1)
    # Mean curves
    max_ep = max(len(h['accuracy']) for h in all_histories)
    mean_train = np.mean([
        np.pad(h['accuracy'], (0, max_ep - len(h['accuracy'])), constant_values=np.nan)
        for h in all_histories
    ], axis=0)
    mean_val = np.mean([
        np.pad(h['val_accuracy'], (0, max_ep - len(h['val_accuracy'])), constant_values=np.nan)
        for h in all_histories
    ], axis=0)
    ax2.plot(mean_train, color='#f5a623', linewidth=2.5, label='Mean Train')
    ax2.plot(mean_val, color='#3498db', linewidth=2.5, label='Mean Val')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Learning Curves (All Folds)', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # 3) Aggregated confusion matrix (normalized)
    ax3 = fig.add_subplot(2, 3, 3)
    agg_cm = np.sum([np.array(m['confusion_matrix']) for m in all_metrics], axis=0)
    cm_norm = agg_cm.astype(float) / agg_cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=ax3,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'shrink': 0.8})
    ax3.set_title('Aggregated Confusion Matrix', fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=11)
    ax3.set_ylabel('True', fontsize=11)

    # 4) Per-class F1-score comparison across folds
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(len(CLASS_NAMES))
    width = 0.15
    for i, m in enumerate(all_metrics):
        f1s = [m['per_class'][cls]['f1'] for cls in CLASS_NAMES]
        ax4.bar(x + i * width, f1s, width, label=f'Fold {m["fold"]}', alpha=0.8)
    # Mean bars
    mean_f1s = [np.mean([m['per_class'][cls]['f1'] for m in all_metrics]) for cls in CLASS_NAMES]
    ax4.bar(x + len(all_metrics) * width, mean_f1s, width,
            label='Mean', color='#e74c3c', edgecolor='#c0392b')
    ax4.set_xlabel('Class', fontsize=11)
    ax4.set_ylabel('F1-Score', fontsize=11)
    ax4.set_title('Per-Class F1 Score', fontweight='bold')
    ax4.set_xticks(x + width * len(all_metrics) / 2)
    ax4.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax4.legend(fontsize=9, ncol=3)
    ax4.grid(axis='y', alpha=0.3)

    # 5) Metrics radar chart
    ax5 = fig.add_subplot(2, 3, 5, polar=True)
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro',
                       'f1_macro', 'kappa', 'miou']
    labels_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa', 'mIoU']
    means = [np.mean([m[k] for m in all_metrics]) for k in metrics_to_plot]
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    means += means[:1]
    angles += angles[:1]
    ax5.fill(angles, means, color='#3498db', alpha=0.25)
    ax5.plot(angles, means, color='#3498db', linewidth=2, marker='o', markersize=6)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(labels_radar, fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.set_title('Model Performance Radar', fontweight='bold', pad=20)

    # 6) Box plot of key metrics
    ax6 = fig.add_subplot(2, 3, 6)
    box_data = {
        'Accuracy': [m['accuracy'] for m in all_metrics],
        'F1 (macro)': [m['f1_macro'] for m in all_metrics],
        'Kappa': [m['kappa'] for m in all_metrics],
        'mIoU': [m['miou'] for m in all_metrics],
    }
    bp = ax6.boxplot(box_data.values(), labels=box_data.keys(), patch_artist=True,
                     boxprops=dict(facecolor='#3498db', alpha=0.6),
                     medianprops=dict(color='#e74c3c', linewidth=2))
    ax6.set_title('Metric Distribution Across Folds', fontweight='bold')
    ax6.set_ylabel('Score', fontsize=11)
    ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, f'cv_{model_type}_{method}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  📊 CV plots saved to: {out_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Research-grade cross-validation for PFZ models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CV Methods:
  temporal_block  Recommended for spatiotemporal data (prevents data leakage)
  walk_forward    Train on past, validate on future (realistic for forecasting)
  stratified      Balanced class distribution per fold (secondary validation)

Example (for research paper):
  python cross_validate.py --data_dir ../backend/uploads/DATA --method temporal_block --k 5 --epochs 5 --model unet
  python cross_validate.py --data_dir ../backend/uploads/DATA --method walk_forward --k 5 --epochs 5 --model convlstm
        """)
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--method',     type=str, default='temporal_block',
                        choices=['temporal_block', 'walk_forward', 'stratified'])
    parser.add_argument('--k',          type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs',     type=int, default=5, help='Epochs per fold')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model',      type=str, default='unet',
                        choices=['unet', 'convlstm'])
    parser.add_argument('--output_dir', type=str, default='../backend/saved_models')
    args = parser.parse_args()

    run_cv(
        data_dir=args.data_dir,
        method=args.method,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model,
        output_dir=args.output_dir
    )
