"""
models/tensorflow_models.py
============================
Model architectures for TensorFlow models.
Used as fallback when loading saved models fails.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def create_cnn_2d_model(num_classes=10):
    """Create the 2D CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=(5, 4, 1)),
        
        layers.Conv2D(16, (2, 2), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Conv2D(64, (2, 2), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.5),
        
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.4),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_lstm_model(num_classes=10):
    """Create the LSTM architecture."""
    model = models.Sequential([
        layers.Input(shape=(1, 19)),
        
        layers.LSTM(64, return_sequences=True, activation='tanh',
                   kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.LSTM(128, return_sequences=False, activation='tanh',
                   kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_transformer_model(num_classes=10):
    """Create the Transformer architecture."""
    
    def transformer_block(x):
        residual = layers.Dense(32)(x)
        x_norm = layers.LayerNormalization()(residual)
        
        attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.3)(x_norm, x_norm)
        x = layers.Add()([residual, attn])
        x = layers.LayerNormalization()(x)
        
        ff = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        ff = layers.Dropout(0.3)(ff)
        ff = layers.Dense(32)(ff)
        
        x = layers.Add()([x, ff])
        return layers.LayerNormalization()(x)
    
    inputs = layers.Input(shape=(19, 1))
    
    x = layers.GaussianNoise(0.02)(inputs)
    x = layers.Dense(32, activation="relu")(x)
    
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model