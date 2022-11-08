""" """

import streamlit as st


def run_app():
    st.title("Dog Identifier App", anchor=None)

    st.subheader(
        "An app to tell you which breed your dog is or which dog you most resemble.",
        anchor=None,
    )

    body = """
    Have you ever taken a picture of a cute dog you saw on the streets and wondered 
    what breed it might be? Now you can easily identify the dogs breed by uploading 
    an image of the dog below and we'll tell you what breed it might be. 
    
    Cool, right? But it even better: You can also upload a picture of a person and 
    we will tell you which dog breed this person most resembles.
     
    Try it out, just upload an image below and see for yourself!
    """
    st.text(body)
    uploaded_image = st.file_uploader(
        "Upload your image here",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Upload an image of a person or a dog",
    )

    if uploaded_image:
        st.image(uploaded_image)


if __name__ == "__main__":
    run_app()
