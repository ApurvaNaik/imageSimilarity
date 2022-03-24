import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import cv2
import os
import urllib.request
import tensorflow as tf
import streamlit as st

from PIL import Image
from pathlib import Path
from scipy import spatial
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model

model_path = 'model/'

@st.cache(allow_output_mutation=True)
def load_vgg19():
    vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    )
    
    vgg19.load_weights(model_path+'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
    # remove the last, prediction layer
    basemodel = Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)
    print(basemodel.summary())
    return basemodel

# read image

def read_image(path):
    img = cv2.imread(path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img


def upload_from_url(url):
    try:
        file_name = os.path.basename(url)
        urllib.request.urlretrieve(url, file_name)
        img = Image.open(file_name)
    except:
        st.error('enter valid URL!')
    return img


st.title("Calculate Image Similarity")
st.write("""Given 2 images compute the image similarity between them. Use the VGG-19
model for generating the feature vector. Create an API that takes the image and returns
the similarity score between them.""")

basemodel = load_vgg19()

# function to get feature vector of the image
def get_similarity_score(image1, image2, basemodel = basemodel):
    rgb_img1 = cv2.cvtColor(np.float32(image1), cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(np.float32(image2), cv2.COLOR_BGR2RGB)
    reshaped_1 = cv2.resize(rgb_img1, (224, 224)).reshape(1, 224, 224, 3)
    reshaped_2 = cv2.resize(rgb_img2, (224, 224)).reshape(1, 224, 224, 3)
    feature_vector_1 = basemodel.predict(reshaped_1)
    feature_vector_2 = basemodel.predict(reshaped_2)
    cosine_similarity = 1 - spatial.distance.cosine(feature_vector_1, feature_vector_2)
    return round(cosine_similarity, 3)

option = st.selectbox(
     'How would you like upload images?',
     ('Upload from device', 'Fetch from URL'))

st.write(option)

if option == 'Upload from device':
    upload1 = st.file_uploader('Upload first image')
    if upload1 is None:
        st.error('Please upload an image!')
    else:
        image1 = Image.open(upload1)

    upload2 = st.file_uploader('Upload second image')
    if upload2 is None:
        st.error('Please upload an image!')
    else:
        image2 = Image.open(upload2)
else: 
    url1 = st.text_input('Upload first image URL', 'https://media.geeksforgeeks.org/wp-content/uploads/20210318103632/gfg-300x300.png')
    if url1 is None: 
        st.error('Please upload a valid image URL!')
    else:
        image1 = upload_from_url(url1)

    url2 = st.text_input('Upload second image URL', 'https://res.cloudinary.com/demo/image/upload/ar_1.0,c_thumb,g_face,w_0.6,z_0.7/r_max/co_black,e_outline/co_grey,e_shadow,x_40,y_55/actor.png')
    if url2 is None:
        st.error('Please upload a valid image URL!')
    else:
        image2 = upload_from_url(url2)

show_images = st.checkbox('show_images')
if show_images:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Image 1")
        st.image(image1, width = 200)

    with col2:
        st.header("Image 2")
        st.image(image2, width = 200)

similarity = st.checkbox('calculate similarity')
if similarity:
    basemodel = load_vgg19()
    sim = get_similarity_score(image1, image2, basemodel)
    st.write('Image Similarity: ', sim)

interpret_score= st.checkbox('interpret score')
if interpret_score and similarity:
    if sim >= 0.7:
        st.write('Images are fairly similar')
    else:
        st.write('Images are not similar')
else:
    st.write('First calculate similarity')
