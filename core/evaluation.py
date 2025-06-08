import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from core.preprocessing import get_preprocessing_function
from config import BASE_MODEL_NAME, THRESHOLD

def evaluate_model(model, x_test, y_test):
    print(f"📊 Evaluating with threshold {THRESHOLD:.2f}...")
    preprocess_input = get_preprocessing_function(BASE_MODEL_NAME)
    x_test = preprocess_input(x_test.astype(np.float32))
    probs = model.predict(x_test)
    predictions = (probs >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)

    print(f"✅ Accuracy: {accuracy:.4f} — F1 Score: {f1:.4f}")
    print(f"🧩 Confusion matrix:\n{confusion}")
    