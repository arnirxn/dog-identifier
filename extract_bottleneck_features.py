"""Utility to make predictions on tensors using Keras Applications."""

from tensorflow.keras.applications import inception_v3, resnet50, vgg16, vgg19, xception


def extract_VGG16(tensor):
    return vgg16.VGG16(weights='imagenet', include_top=False).predict(vgg16.preprocess_input(tensor))


def extract_VGG19(tensor):
    return vgg19.VGG19(weights='imagenet', include_top=False).predict(vgg19.preprocess_input(tensor))


def extract_Resnet50(tensor):
    return resnet50.ResNet50(weights='imagenet', include_top=False).predict(resnet50.preprocess_input(tensor))


def extract_Xception(tensor):
    return xception.Xception(weights='imagenet', include_top=False).predict(xception.preprocess_input(tensor))


def extract_InceptionV3(tensor):
    return inception_v3.InceptionV3(weights='imagenet', include_top=False).predict(inception_v3.preprocess_input(tensor))
