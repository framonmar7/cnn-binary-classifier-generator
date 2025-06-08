import os
from datetime import datetime

# ============================
# 📁 System paths
# ============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

TRAIN_POS_DIR = os.path.join(DATA_DIR, 'train', 'positive')
TRAIN_NEG_DIR = os.path.join(DATA_DIR, 'train', 'negative')
TEST_POS_DIR = os.path.join(DATA_DIR, 'test', 'positive')
TEST_NEG_DIR = os.path.join(DATA_DIR, 'test', 'negative')

MODEL_FILENAME = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
MODEL_PATH = os.path.join(BASE_DIR, 'models', MODEL_FILENAME)

# ============================
# 🖼️ Image parameters
# ============================

IMG_SIZE = 224  # Input images will be resized to (IMG_SIZE, IMG_SIZE, 3)
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# ============================
# 🔁 Data augmentation settings
# ============================

AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.12,
    "height_shift_range": 0.12,
    "shear_range": 8,
    "zoom_range": [0.9, 1.2],
    "brightness_range": [0.8, 1.2],
    "horizontal_flip": True,
    "vertical_flip": False,
    "fill_mode": 'nearest'
}

# ============================
# 🧠 Model architecture
# ============================

BASE_MODEL_NAME = "ResNet50"

CLASSIFIER_HEAD = [
    {"type": "GlobalAveragePooling2D"},
    {"type": "Dense", "units": 256, "activation": "relu"},
    {"type": "Dropout", "rate": 0.5},
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "Dropout", "rate": 0.3},
    {"type": "Dense", "units": 1, "activation": "sigmoid"}
]

# ============================
# 🏋️‍♂️ Training parameters
# ============================

BATCH_SIZE = 16
INITIAL_EPOCHS = 35
FINE_TUNE_EPOCHS = 15
UNFROZEN_LAYERS = 50

LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINE_TUNE = 5e-6

# ============================
# ⏱️ Callbacks configuration
# ============================

EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

# ============================
# 📊 Evaluation parameters
# ============================

THRESHOLD = 0.5

# ============================
# 🧾 Logging
# ============================

VERBOSE = 2
