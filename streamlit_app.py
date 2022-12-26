# pip install tensorflow

import streamlit as st
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform

st.title('Shoe Prediction')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    np_image = np.array(img).astype('float32')
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    
    dict = {0: 'Ballet Flat',
    1: 'Boat',
    2: 'Brogue',
    3: 'Clog',
    4: 'Sneaker'}
    
    model = load_model('eff_b0_shoe.h5')
    y_pred = model.predict(np_image)
    y_class = [np.argmax(element) for element in y_pred]
    res = dict[y_class[0]], "- Confidence:", y_pred[0][y_class[0]]*100
    st.success(res)