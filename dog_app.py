""" """

from pathlib import Path
import streamlit as st
from PIL import Image

from app_utils import dog_detected_Resnet50, get_faces, predict_breed_with_Xception


def display_welcome():
    """Display title, subheader and body message of the app."""
    st.title("Dog Identifier App", anchor=None)

    st.subheader(
        "An app to tell you which breed your dog is or which dog you most resemble.",
        anchor=None,
    )

    st.text(
        """
        Have you ever taken a picture of a cute dog you saw on the streets and wondered 
        what breed it might be? Now you can easily identify the dogs breed by uploading 
        an image of the dog below and we'll tell you what breed it might be. 

        Cool, right? But it even better: You can also upload a picture of a person and 
        we will tell you which dog breed this person most resembles.

        Try it out, just upload an image below and see for yourself!
        """
    )

    st.write("")


def run_app():

    uploaded_image = st.file_uploader(
        "Upload your image here",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Upload an image of a dog or a person.",
    )

    if uploaded_image:
        image_path = str(Path("app", "temp", uploaded_image.name))
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Get faces
        num_faces = len(get_faces(image_path))

        # Check if dogs are detected
        dog_detected = dog_detected_Resnet50(image_path)

        # Check more than one face is detected
        if num_faces > 1:
            st.warning(
                "More than one face was detected in your image. Please only upload an image with one face in it."
            )

        # Check if any dog or face was detected
        elif not num_faces and not dog_detected:
            st.warning(
                "We couldn't find a dog, nor a face in your image. Please upload an image containing a face or a dog."
            )

        else:
            st.write("Dog detected:", dog_detected)
            st.write("Faces detected:", num_faces)

            # Predict dog breed using Xception
            st.write("Predicting what breed this looks like ...")
            prediction = predict_breed_with_Xception(image_path, verbose=0)

            subject = "dog" if dog_detected else "human"
            subject = "human/dog" if dog_detected and num_faces else subject
            article = "an" if prediction[0].lower() == "a" else "a"
            breed = prediction.replace("_", " ").title()

        #
        # st.image(uploaded_image)
    #
    # if uploaded_file is not None:
    #     u_img = Image.open(uploaded_file)
    #     st.image(image, use_column_width=True)
    #     show.image(u_img, "Uploaded Image", use_column_width=True)


if __name__ == "__main__":
    run_app()
