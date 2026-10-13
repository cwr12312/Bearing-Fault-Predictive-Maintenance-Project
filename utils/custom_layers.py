"""
utils/custom_layers.py
======================
Custom layer implementations to handle deserialization of models
saved with newer TensorFlow versions.
"""

import tensorflow as tf                        # core TF/Keras library
from tensorflow.keras import layers            # base layer classes (Dense, Conv2D) we subclass below
from tensorflow.keras import regularizers      # imported for availability when reconstructing layer configs (not directly used below)
from tensorflow.keras import initializers      # imported for availability when reconstructing layer configs (not directly used below)
import numpy as np                              # imported for availability / potential array handling (not directly used below)


class DenseV2(layers.Dense):
    """
    Custom Dense layer that ignores quantization_config parameter
    for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        # Remove quantization_config if present
        kwargs.pop('quantization_config', None)          # strip an arg newer Keras adds that older Dense doesn't accept
        super().__init__(*args, **kwargs)                 # continue normal Dense initialization
    
    @classmethod
    def from_config(cls, config):
        # Remove quantization_config from config
        config.pop('quantization_config', None)           # strip the same key when rebuilding from a saved model's config dict
        return super().from_config(config)                 # delegate to Keras' standard reconstruction


class Conv2DV2(layers.Conv2D):
    """
    Custom Conv2D layer that ignores quantization_config parameter
    for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)            # same fix as DenseV2, applied to Conv2D
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)             # same fix as DenseV2.from_config
        return super().from_config(config)


# Custom objects mapping for deserialization
CUSTOM_LAYERS = {
    'Dense': DenseV2,                                        # swap in the patched Dense when loading a saved model
    'Conv2D': Conv2DV2,                                       # swap in the patched Conv2D when loading a saved model
    'DTypePolicy': tf.keras.mixed_precision.Policy,           # resolves the mixed-precision policy class by name
    'GlorotUniform': tf.keras.initializers.GlorotUniform,     # resolves this initializer class by name
    'Zeros': tf.keras.initializers.Zeros,                     # resolves this initializer class by name
}


# Register custom objects with Keras
for name, layer in CUSTOM_LAYERS.items():                     # loop over each name -> class mapping above
    tf.keras.utils.get_custom_objects()[name] = layer         # register it globally so Keras' loader can resolve these names in saved model files
