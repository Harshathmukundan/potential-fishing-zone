"""
train.py — Full Training Pipeline
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N

Usage:
    python train.py --data_dir /path/to/nc_files --epochs 5
    python train.py --data_dir ./data --epochs 5 --batch_size 4
    python train.py --data_dir ./data --skip_unet   # Only train ConvLSTM
    python train.py --data_dir ./data --skip_convlstm  # Only train U-Net
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import mixed_precision # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

from unet     import build_unet
from convlstm import build_convlstm
from dataset  import load_datasets, build_sample, make_label, PFZGenerator, ConvLSTMGenerator

# ─── Config ──────────────────────────────────────────────────
SEED       = 42
TARGET     = 256
LAT_MIN, LAT_MAX = 5.0,  14.5
LON_MIN, LON_MAX = 77.0, 84.92

np.random.seed(SEED)
tf.random.set_seed(SEED)
mixed_precision.set_global_policy('mixed_float16')


def train(data_dir, epochs=5, batch_size=4, output_dir='../backend/saved_models',
         skip_unet=False, skip_convlstm=False):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    das, N_TIMES = load_datasets(data_dir)

    # ── Split ─────────────────────────────────────────────────
    print("[2/5] Splitting data...")
    all_idx = np.arange(1, N_TIMES - 1)
    np.random.shuffle(all_idx)
    split     = int(0.8 * len(all_idx))
    train_idx = all_idx[:split]
    val_idx   = all_idx[split:]

    train_gen = PFZGenerator(das, train_idx, N_TIMES, batch_size=batch_size)
    val_gen   = PFZGenerator(das, val_idx,   N_TIMES, batch_size=batch_size, shuffle=False)

    print(f"   Train batches: {len(train_gen)} | Val batches: {len(val_gen)}")

    # ── U-Net ─────────────────────────────────────────────────
    unet_history = None
    if skip_unet:
        print("\n[3/5] Skipping U-Net (--skip_unet flag set)")
        existing = os.path.join(output_dir, 'best_unet.keras')
        if os.path.exists(existing):
            print(f"   Using existing model: {existing}")
    else:
        print("\n[3/5] Training U-Net...")
        unet = build_unet(input_channels=24)
        unet.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), # type: ignore
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        unet_callbacks = [
            tf.keras.callbacks.EarlyStopping( # type: ignore
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau( # type: ignore
                monitor='val_loss', patience=4,
                factor=0.5, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint( # type: ignore
                os.path.join(output_dir, 'best_unet.keras'),
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]

        unet_history = unet.fit(
            train_gen, validation_data=val_gen,
            epochs=epochs, callbacks=unet_callbacks
        )
        unet.save(os.path.join(output_dir, 'unet_final.keras'))

    # ── ConvLSTM ──────────────────────────────────────────────
    lstm_history = None
    if skip_convlstm:
        print("\n[4/5] Skipping ConvLSTM (--skip_convlstm flag set)")
        existing = os.path.join(output_dir, 'best_convlstm.keras')
        if os.path.exists(existing):
            print(f"   Using existing model: {existing}")
    else:
        print("\n[4/5] Training ConvLSTM...")
        lstm_train = ConvLSTMGenerator(das, train_idx, N_TIMES, batch_size=batch_size)
        lstm_val   = ConvLSTMGenerator(das, val_idx,   N_TIMES, batch_size=batch_size)

        clstm = build_convlstm()
        clstm.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), # type: ignore
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        lstm_callbacks = [
            tf.keras.callbacks.EarlyStopping( # type: ignore
                monitor='val_accuracy', patience=8,
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau( # type: ignore
                monitor='val_loss', patience=4,
                factor=0.5, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint( # type: ignore
                os.path.join(output_dir, 'best_convlstm.keras'),
                monitor='val_accuracy', save_best_only=True, verbose=1
            )
        ]

        lstm_history = clstm.fit(
            lstm_train, validation_data=lstm_val,
            epochs=epochs, callbacks=lstm_callbacks
        )
        clstm.save(os.path.join(output_dir, 'convlstm_final.keras'))

    # ── Save training history JSON ────────────────────────────
    history_data = {}
    if unet_history:
        history_data['unet'] = {
            'accuracy':     unet_history.history['accuracy'],
            'val_accuracy': unet_history.history['val_accuracy'],
            'loss':         unet_history.history['loss'],
            'val_loss':     unet_history.history['val_loss'],
        }
    if lstm_history:
        history_data['convlstm'] = {
            'accuracy':     lstm_history.history['accuracy'],
            'val_accuracy': lstm_history.history['val_accuracy'],
            'loss':         lstm_history.history['loss'],
            'val_loss':     lstm_history.history['val_loss'],
        }

    # Merge with existing history if available
    hist_path = os.path.join(output_dir, 'training_history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            existing = json.load(f)
        existing.update(history_data)
        history_data = existing

    if history_data:
        with open(hist_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"   Saved training history to {hist_path}")

    # ── Evaluation plots ──────────────────────────────────────
    print("\n[5/5] Generating evaluation plots...")
    if unet_history and lstm_history:
        _plot_training(unet_history, lstm_history, output_dir)
        _plot_confusion(unet, val_gen, output_dir)
    elif unet_history:
        _plot_confusion(unet, val_gen, output_dir)

    print(f"\n✅ Training complete!")
    if unet_history:
        best_unet_acc = max(unet_history.history['val_accuracy'])
        print(f"   U-Net best val accuracy   : {best_unet_acc*100:.2f}%")
    if lstm_history:
        best_lstm_acc = max(lstm_history.history['val_accuracy'])
        print(f"   ConvLSTM best val accuracy: {best_lstm_acc*100:.2f}%")
    print(f"   Models saved to: {output_dir}")


def _plot_training(unet_h, lstm_h, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax00, ax01 = axes[0][0], axes[0][1] # type: ignore
    ax10, ax11 = axes[1][0], axes[1][1] # type: ignore
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    for ax, hist, title in zip(
        [ax00, ax01], [unet_h, lstm_h], ['U-Net', 'ConvLSTM']
    ):
        ax.plot(hist.history['loss'],     label='Train', color='#e74c3c', lw=2)
        ax.plot(hist.history['val_loss'], label='Val',   color='#3498db', lw=2, ls='--')
        ax.set_title(f'{title} Loss');  ax.legend(); ax.grid(alpha=0.3)

    for ax, hist, title in zip(
        [ax10, ax11], [unet_h, lstm_h], ['U-Net', 'ConvLSTM']
    ):
        ax.plot(hist.history['accuracy'],     label='Train', color='#e74c3c', lw=2)
        ax.plot(hist.history['val_accuracy'], label='Val',   color='#3498db', lw=2, ls='--')
        ax.axhline(0.90, color='green', ls=':', lw=1.5, label='90% target')
        ax.set_title(f'{title} Accuracy'); ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()


def _plot_confusion(model, val_gen, output_dir):
    preds, trues = [], []
    for i in range(min(4, len(val_gen))):
        X, Y = val_gen[i]
        p    = model.predict(X, verbose=0)
        preds.append(np.argmax(p, axis=-1))
        trues.append(np.argmax(Y, axis=-1))

    pred_flat = np.concatenate(preds).flatten()
    true_flat = np.concatenate(trues).flatten()

    cm = confusion_matrix(true_flat, pred_flat, normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6))
    
    labels = ['Low','Medium','High']
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                xticklabels=labels, # type: ignore
                yticklabels=labels) # type: ignore
    ax.set_title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    print(classification_report(true_flat, pred_flat,
                                 target_names=['Low','Medium','High']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PFZ models')
    parser.add_argument('--data_dir',       type=str, required=True)
    parser.add_argument('--epochs',         type=int, default=5)
    parser.add_argument('--batch_size',     type=int, default=4)
    parser.add_argument('--output_dir',     type=str, default='../backend/saved_models')
    parser.add_argument('--skip_unet',      action='store_true', help='Skip U-Net training')
    parser.add_argument('--skip_convlstm',  action='store_true', help='Skip ConvLSTM training')
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.output_dir,
          skip_unet=args.skip_unet, skip_convlstm=args.skip_convlstm)
