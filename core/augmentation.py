from tensorflow.keras.preprocessing.image import ImageDataGenerator
from core.preprocessing import get_preprocessing_function
from config import AUGMENTATION_CONFIG, BASE_MODEL_NAME

def create_dynamic_aug_datagen():
    preprocess_input = get_preprocessing_function(BASE_MODEL_NAME)
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        **AUGMENTATION_CONFIG
    )
