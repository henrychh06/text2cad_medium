# cad_processing/utils/macro.py
# Token constants
N_BIT = 8  # Default quantization bit depth
END_PAD = 5  # End token value
BOOLEAN_PAD = 7  # Boolean operation token value
PRECISION = 1e-5  # Precision for geometric operations

# Token types for the sketch sequence
SKETCH_TOKEN = [
    "PAD",
    "CLS",
    "END",
    "END_SKETCH",
    "END_FACE",
    "END_LOOP",
    "END_CURVE",
]