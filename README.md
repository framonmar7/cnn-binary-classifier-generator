
# 🧠 CNN Binary Classifier Generator

This project is a modular and configurable pipeline to train binary image classifiers using Convolutional Neural Networks (CNNs) with transfer learning.  
This tool is intended for researchers, students, or developers who need to quickly create custom classifiers by simply dropping labeled images into folders and running a single command.

---

## 🚀 Features

- Clean modular architecture based on Python and TensorFlow.
- Uses `MobileNetV2` (or other Keras models) as backbone.
- Custom classification head defined via configuration.
- Data augmentation and fine-tuning support.
- Easy-to-use training and evaluation workflow.
- Reproducibility and portability via `config.py`.

---

## 📦 Requirements

Install the dependencies in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
cnn-classifier/
│
├── core/                # Core logic (dataset, training, evaluation, model...)
├── data/                # Training and testing images
│   ├── train/
│   │   ├── positive/    # Class 1 (e.g., happy face)
│   │   └── negative/    # Class 0 (e.g., sad face)
│   └── test/
│       ├── positive/
│       └── negative/
│
├── models/              # Trained models will be saved here
├── config.py            # Central configuration
├── run.py               # Main entry point
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚡ Quick Start (Default Setup)

To run the training and evaluation using the default configuration:

```bash
python run.py
```

This will:

- Load the training and testing datasets from `data/train/` and `data/test/`.
- Train a binary image classifier with transfer learning.
- Fine-tune the last N layers of the base model.
- Evaluate it using accuracy, F1 score, and a confusion matrix.
- Save the trained model in the `models/` folder.

---

## ⚙️ Advanced Usage (Custom Configuration)

You can fully control the pipeline through the `config.py` file. Key options include:

### 🏗 Model Architecture

```python
BASE_MODEL_NAME = "MobileNetV2"
CLASSIFIER_HEAD = [
    {"type": "GlobalAveragePooling2D"},
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "Dropout", "rate": 0.5},
    {"type": "Dense", "units": 64, "activation": "relu"},
    {"type": "Dropout", "rate": 0.3},
    {"type": "Dense", "units": 1, "activation": "sigmoid"}
]
```

> ✅ **Note:** The list of supported base models is defined in `core/preprocessing.py`.  
> These include popular architectures from `tensorflow.keras.applications`, such as `MobileNetV2`, `ResNet50`, `VGG16`, etc.

### 🖼 Image Processing

```python
IMG_SIZE = 128
AUGMENTATION_CONFIG = {
    "rotation_range": 25,
    "zoom_range": [0.85, 1.3],
    "horizontal_flip": True
}
```

### 🧠 Training Parameters

```python
INITIAL_EPOCHS = 40
FINE_TUNE_EPOCHS = 20
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINE_TUNE = 1e-5
UNFROZEN_LAYERS = 20
```

### 📊 Evaluation

```python
THRESHOLD = 0.5
```

---

## 💾 Output

After training, a model file (e.g., `happiness_classifier_model.keras`) will be saved in the `models/` directory.  
You can load it later for inference or evaluation:

```python
from tensorflow.keras.models import load_model
model = load_model("models/happiness_classifier_model.keras")
```

---

## 🧪 Example: Training Will Smith Classifier

Prepare your dataset like this:

```
data/train/positive/  →  Images of Will Smith
data/train/negative/  →  Images of other people
data/test/positive/   →  Will Smith (test set)
data/test/negative/   →  Other people (test set)
```

Then run:

```bash
python run.py
```

You’ll get output like:

```
✅ Accuracy: 0.9200 — F1 Score: 0.9231
🧩 Confusion matrix:
[[44  6]
 [ 2 48]]
```

---

## 📌 Notes

- The `models/` and `data/` folders are tracked but empty. You should place your own data and trained models there.
- These folders contain `.gitkeep` files to preserve structure in Git.

---

## 📜 License

This project is released under the [MIT License](LICENSE).  
You are free to use, modify, and distribute it — with attribution.

---

## 👤 Author

Developed by [Francisco Jesús Montero Martínez](https://github.com/framonmar7)  
For suggestions, improvements, or collaboration, feel free to reach out.
