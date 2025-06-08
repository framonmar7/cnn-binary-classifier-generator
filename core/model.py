import tensorflow as tf
import inspect
from config import IMG_SIZE, BASE_MODEL_NAME, CLASSIFIER_HEAD, LEARNING_RATE_INITIAL
from core.preprocessing import SUPPORTED_MODELS

def get_base_model():
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    if BASE_MODEL_NAME not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported base model: {BASE_MODEL_NAME}. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS.keys())}"
        )

    model_fn = getattr(tf.keras.applications, BASE_MODEL_NAME)
    base_model = model_fn(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    return base_model

def build_classifier_head(config):
    head = []
    for layer_def in config:
        layer_type = layer_def["type"]
        kwargs = {k: v for k, v in layer_def.items() if k != "type"}

        if not hasattr(tf.keras.layers, layer_type):
            raise ValueError(f"Unsupported layer type: {layer_type}")

        layer_class = getattr(tf.keras.layers, layer_type)
        try:
            validate_layer_params(layer_class, kwargs)
            layer = layer_class(**kwargs)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for layer {layer_type}: {e}")

        head.append(layer)

    return head

def validate_layer_params(layer_class, kwargs):
    sig = inspect.signature(layer_class.__init__)
    required = [
        name for name, param in sig.parameters.items()
        if name != 'self' and param.default is param.empty and param.kind != param.VAR_KEYWORD
    ]
    missing = [r for r in required if r not in kwargs]

    if missing:
        raise ValueError(
            f"Missing required parameters for {layer_class.__name__}: {missing}"
        )

def build_model():
    base_model = get_base_model()
    head = build_classifier_head(CLASSIFIER_HEAD)

    model = tf.keras.Sequential([base_model] + head)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model
