import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

set_background('Streamlit/bg.jpg')


# set title
st.title('plant disease classification')

# set header
st.header('Please upload an image of plant leaves')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('Streamlit/Mobilenetv3large_paddy_disease_detection_architecture_3_fine_tuned.keras')

# load class names
with open('Streamlit/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))