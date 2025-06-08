import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from core.preprocessing import get_preprocessing_function
from config import (
    INITIAL_EPOCHS, FINE_TUNE_EPOCHS, BATCH_SIZE,
    UNFROZEN_LAYERS, VERBOSE, BASE_MODEL_NAME,
    LEARNING_RATE_FINE_TUNE, EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_LR
)

def get_callbacks():
    return [
        EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=VERBOSE),
        ReduceLROnPlateau(factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=VERBOSE)
    ]

def train_head(model, datagen, x_train, y_train, x_val, y_val):
    print("🔧 Training classification head...")
    preprocess_input = get_preprocessing_function(BASE_MODEL_NAME)
    model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=INITIAL_EPOCHS,
        validation_data=(preprocess_input(x_val.astype(np.float32)), y_val),
        steps_per_epoch=int(np.ceil(len(x_train) / BATCH_SIZE)),
        callbacks=get_callbacks(),
        verbose=VERBOSE
    )

def fine_tune(model, datagen, x_train, y_train, x_val, y_val):
    print(f"🎯 Fine-tuning last {UNFROZEN_LAYERS} layers of base model...")
    preprocess_input = get_preprocessing_function(BASE_MODEL_NAME)

    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-UNFROZEN_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=FINE_TUNE_EPOCHS,
        validation_data=(preprocess_input(x_val.astype(np.float32)), y_val),
        steps_per_epoch=int(np.ceil(len(x_train) / BATCH_SIZE)),
        callbacks=get_callbacks(),
        verbose=VERBOSE
    )

def train_model(model, datagen, train_images, train_labels, test_images, test_labels):
    train_head(model, datagen, train_images, train_labels, test_images, test_labels)
    fine_tune(model, datagen, train_images, train_labels, test_images, test_labels)
    