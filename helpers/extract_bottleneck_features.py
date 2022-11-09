"""Utility to make predictions on tensors using Keras Applications."""

from tensorflow.keras.applications import inception_v3, resnet50, vgg16, vgg19, xception


def extract_VGG16(tensor, verbose=1):
    return vgg16.VGG16(weights="imagenet", include_top=False).predict(
        vgg16.preprocess_input(tensor), verbose=verbose
    )


def extract_VGG19(tensor, verbose=1):
    return vgg19.VGG19(weights="imagenet", include_top=False).predict(
        vgg19.preprocess_input(tensor), verbose=verbose
    )


def extract_Resnet50(tensor, verbose=1):
    return resnet50.ResNet50(weights="imagenet", include_top=False).predict(
        resnet50.preprocess_input(tensor), verbose=verbose
    )


def extract_Xception(tensor, verbose=1):
    return xception.Xception(weights="imagenet", include_top=False).predict(
        xception.preprocess_input(tensor), verbose=verbose
    )


def extract_InceptionV3(tensor, verbose=1):
    return inception_v3.InceptionV3(weights="imagenet", include_top=False).predict(
        inception_v3.preprocess_input(tensor), verbose=verbose
    )
