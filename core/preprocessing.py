from tensorflow.keras.applications import (
    mobilenet_v2, mobilenet_v3,
    efficientnet, resnet, densenet,
    inception_v3, xception, nasnet,
    vgg16, vgg19
)

SUPPORTED_MODELS = {
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "MobileNetV3Small": mobilenet_v3.preprocess_input,
    "MobileNetV3Large": mobilenet_v3.preprocess_input,
    "EfficientNetB0": efficientnet.preprocess_input,
    "EfficientNetB1": efficientnet.preprocess_input,
    "ResNet50": resnet.preprocess_input,
    "DenseNet121": densenet.preprocess_input,
    "InceptionV3": inception_v3.preprocess_input,
    "Xception": xception.preprocess_input,
    "NASNetMobile": nasnet.preprocess_input,
    "VGG16": vgg16.preprocess_input,
    "VGG19": vgg19.preprocess_input,
}

def get_preprocessing_function(model_name):
    try:
        return SUPPORTED_MODELS[model_name]
    except KeyError:
        raise ValueError(
            f"Unsupported model for preprocessing: {model_name}. "
            f"Supported models are: {', '.join(SUPPORTED_MODELS.keys())}"
        )
