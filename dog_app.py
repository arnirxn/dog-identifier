"""A streamlit web app to allow users to get their own pictures classified."""

from pathlib import Path

import streamlit as st

from helpers.app_utils import (
    dog_detected_Resnet50,
    get_faces,
    predict_breed_with_Xception,
    get_model,
)


def display_welcome():
    """Display title, subheader and body message of the app."""
    st.title("Dog Identifier App", anchor=None)

    st.subheader(
        "Find out which breed a dog or person resembles most",
        anchor=None,
    )

    st.text(
        """
        Have you ever taken a picture of a cute dog you saw on the streets and wondered 
        what breed it might be? Now you can easily identify the dogs breed by uploading 
        an image of the dog below and we'll tell you what breed it might be. 

        Cool, right? But it even better: You can also upload a picture of a person and 
        we will tell you which dog breed this person most resembles.

        Try it out, just upload an image of a dog or person below and see for yourself!
        """
    )

    st.write("")


def run_app():
    """Run the streamlit web app."""

    display_welcome()

    uploaded_image = st.file_uploader(
        " ",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Upload an image of a dog or a person.",
    )

    if uploaded_image:

        dir_path = Path(".temp")
        image_path = str(dir_path / uploaded_image.name)

        # Create directory
        dir_path.mkdir(parents=True, exist_ok=True)

        # Write image locally
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
            with st.spinner(text="Predicting what breed this looks like ..."):

                # Get Xception model
                Xception_model = get_model()

                # Predict dog breed using Xception
                prediction = predict_breed_with_Xception(
                    image_path, model=Xception_model, verbose=0
                )

                subject = "dog" if dog_detected else "human"
                subject = "human/dog" if dog_detected and num_faces else subject
                article = "an" if prediction[0].lower() == "a" else "a"
                breed = prediction.replace("_", " ").title()

            st.success(f"This {subject} picture looks like {article} {breed}!")
            st.image(uploaded_image, use_column_width=True)


if __name__ == "__main__":
    run_app()
