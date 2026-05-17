"""
unet.py — U-Net Architecture for PFZ Spatial Segmentation
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N
"""

import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore


def conv_block(x, filters, dropout_rate=0.1):
    """Double Conv2D + BatchNorm + optional SpatialDropout."""
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def build_unet(input_channels=24, target=256, num_classes=3):
    """
    U-Net for pixel-wise PFZ segmentation.

    Args:
        input_channels: 8 features × 3 days = 24
        target: spatial resolution (256×256)
        num_classes: Low / Medium / High = 3

    Returns:
        Keras Model
    """
    inp = layers.Input((target, target, input_channels), name='unet_input')

    # ── Encoder ──────────────────────────────────────────────
    c1 = conv_block(inp,  64,  dropout_rate=0.0)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1,  128, dropout_rate=0.1)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = conv_block(p2,  256, dropout_rate=0.1)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = conv_block(p3,  512, dropout_rate=0.2)
    p4 = layers.MaxPooling2D(2)(c4)

    # ── Bottleneck ───────────────────────────────────────────
    bn = conv_block(p4, 1024, dropout_rate=0.3)

    # ── Decoder with skip connections ────────────────────────
    u1 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(bn)
    u1 = layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 512, dropout_rate=0.2)

    u2 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c5)
    u2 = layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 256, dropout_rate=0.1)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c6)
    u3 = layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 128, dropout_rate=0.1)

    u4 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c7)
    u4 = layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 64, dropout_rate=0.0)

    # ── Output ───────────────────────────────────────────────
    out = layers.Conv2D(num_classes, 1, activation='softmax',
                        dtype='float32', name='pfz_output')(c8)

    model = Model(inp, out, name='PFZ_UNet')
    return model


if __name__ == '__main__':
    model = build_unet()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
