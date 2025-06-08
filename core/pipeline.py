from core.dataset import load_dataset
from core.augmentation import create_dynamic_aug_datagen
from core.model import build_model
from core.training import train_model
from core.evaluation import evaluate_model
from config import TRAIN_POS_DIR, TRAIN_NEG_DIR, TEST_POS_DIR, TEST_NEG_DIR, MODEL_PATH

def run():
    try:
        train_images, train_labels = load_dataset(TRAIN_POS_DIR, TRAIN_NEG_DIR)
        test_images, test_labels = load_dataset(TEST_POS_DIR, TEST_NEG_DIR)

        datagen = create_dynamic_aug_datagen()
        datagen.fit(train_images)

        model = build_model()

    except Exception as prep_error:
        print(f"❌ An error occurred during data/model preparation: {prep_error}")
        raise

    try:
        train_model(model, datagen, train_images, train_labels, test_images, test_labels)
        evaluate_model(model, test_images, test_labels)

    except Exception as train_error:
        print(f"❌ An error occurred during training or evaluation: {train_error}")
        raise

    finally:
        try:
            model.save(MODEL_PATH)
            print(f"💾 Model saved on: {MODEL_PATH}")
        except Exception as save_error:
            print(f"⚠️ Failed to save model: {save_error}")
