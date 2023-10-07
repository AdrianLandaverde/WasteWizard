import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from huggingface_hub import from_pretrained_fastai
import pathlib

# Define the CSS style
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://free4kwallpapers.com/uploads/originals/2018/07/19/forest-at-dusk-mikael-gustafsson--download-link-in-comments-wallpaper.jpg");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
# Inject the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


def plastic(inp):
    plastic_model = from_pretrained_fastai("pyesonekyaw/recycletree_plastic")
    return plastic_model.predict(inp)

def paper(inp):
    plastic_model = from_pretrained_fastai("pyesonekyaw/recycletree_paper")
    return plastic_model.predict(inp)

def metal(inp):
    plastic_model = from_pretrained_fastai("pyesonekyaw/recycletree_metal")
    return plastic_model.predict(inp)

def others(inp):
    plastic_model = from_pretrained_fastai("pyesonekyaw/recycletree_others")
    return plastic_model.predict(inp)

def glass(inp):
    plastic_model = from_pretrained_fastai("pyesonekyaw/recycletree_glass")
    return plastic_model.predict(inp)


def model(image):
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    import os

    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    materials_model = from_pretrained_fastai("pyesonekyaw/recycletree_materials")
    return materials_model.predict(image)

st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)



st.title('🌳Waste Wizard🌳')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image

    image = Image.open(uploaded_file)
    img_array = np.array(image)
    print(img_array.shape)
    print(type(img_array))

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image.')

    with col2:
        with st.spinner('## Predicting...'):
            prediction= model(img_array)[0]
            if prediction=="others":
                rec="Not Recyclable"
            else:
                rec="Recyclable"

            if prediction=="plastic":
                tipo= plastic(img_array)[0]
            elif prediction=="glass":
                tipo= glass(img_array[0])
            elif prediction=="paper":
                tipo= paper(img_array)[0]
            elif prediction=="metal":
                tipo= metal(img_array)[0]
            elif prediction=="others":
                tipo= others(img_array)[0]

            st.markdown("# Material \n ### " + prediction)
            st.markdown("# Type \n ### " +tipo)
            st.markdown("# Is Recyclable? \n ### "+ rec)
            st.balloons()

