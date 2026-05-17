"""
convlstm.py — ConvLSTM Architecture for PFZ Temporal Forecasting
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N
"""

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore


def build_convlstm(seq_len=7, n_features=8, target=256, num_classes=3):
    """
    ConvLSTM for 7-day-ahead PFZ forecasting.

    Input:  (seq_len, target, target, n_features)
    Output: (target, target, num_classes)

    Args:
        seq_len:     Number of input days (default 7)
        n_features:  Channels per day (default 8)
        target:      Spatial resolution (default 256)
        num_classes: PFZ classes (default 3)

    Returns:
        Keras Model
    """
    inp = layers.Input(
        (seq_len, target, target, n_features),
        name='convlstm_input'
    )

    # ── Spatiotemporal encoding ───────────────────────────────
    x = layers.ConvLSTM2D(
        64, kernel_size=3, padding='same',
        return_sequences=True,
        kernel_initializer='he_normal',
        recurrent_dropout=0.1,
        name='convlstm_1'
    )(inp)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
        32, kernel_size=3, padding='same',
        return_sequences=False,
        kernel_initializer='he_normal',
        recurrent_dropout=0.1,
        name='convlstm_2'
    )(x)
    x = layers.BatchNormalization()(x)

    # ── Spatial refinement ───────────────────────────────────
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    # ── Output ───────────────────────────────────────────────
    out = layers.Conv2D(
        num_classes, 1, activation='softmax',
        dtype='float32', name='forecast_output'
    )(x)

    model = Model(inp, out, name='PFZ_ConvLSTM')
    return model


if __name__ == '__main__':
    model = build_convlstm()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
