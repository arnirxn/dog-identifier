from pathlib import Path

import cv2
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.applications.xception import preprocess_input, Xception
from tensorflow.keras.utils import img_to_array, load_img


def get_model():
    """
    Builds a sequential model and loads pre-trained Xception weights on it.
    :return: Xception model
    """

    # Define architecture
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Xception_model.add(Dense(133, activation="softmax"))

    # Load the model weights
    Xception_model.load_weights("saved_models/weights.best.Xception.hdf5")

    return Xception_model


def image_to_tensor(img_path):
    """
    Takes a string-valued file path to a color image as input and
    returns a 4D tensor suitable for supplying to a Keras CNN.

    :param str img_path: File path to a color image
    :return: 4D tensor suitable for supplying to a Keras CNN
    """

    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    tensor_4d = np.expand_dims(x, axis=0)

    return tensor_4d


def predict_breed_with_Xception(img_path, model, verbose=1):
    """
    Take a dog image and returns the dog breed predicted by Xception.

    :param str img_path: Path to the dog image
    :return: Name of the predicted breed
    """

    # Transform image to 4-D tensor
    tensor = image_to_tensor(img_path)

    # Extract bottleneck features
    bottleneck_feature = Xception(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor), verbose=0
    )

    # Obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature, verbose=verbose)

    # Get dog breed that is predicted by the model
    with open(Path("helpers", "dog_names.txt"), "r") as f:
        dog_names = f.read().splitlines()

    predicted_dog_breed = dog_names[np.argmax(predicted_vector)]

    return predicted_dog_breed


def get_faces(image_path):
    """
    Takes image and return faces found.

    :param image: BGR Image
    :return: Array of faces detected
    """

    # Extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")

    # Convert BGR image to grayscale
    gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

    # Find faces in image
    faces = face_cascade.detectMultiScale(gray)

    return faces


def dog_detected_Resnet50(image_path):
    """
    Takes an image and returns True if a dog is detected with ResNet50.

    :param str image_path: Path to the image
    :return: True if dog is detected, else, False
    :rtype: bool
    """

    img = preprocess_input(image_to_tensor(image_path))

    # define ResNet50 model
    ResNet50_model = ResNet50(weights="imagenet")

    # returns prediction vector for image located at img_path
    prediction = np.argmax(ResNet50_model.predict(img, verbose=0))

    dog_detected = (prediction <= 268) & (prediction >= 151)

    return dog_detected
