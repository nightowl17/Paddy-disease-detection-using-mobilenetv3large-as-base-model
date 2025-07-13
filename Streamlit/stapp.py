import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from pathlib import Path
import keras
from util import classify, set_background



# set title
st.title('plant disease classification')

# set header
st.header('Please upload an image of plant leaves')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


model_path = Path(__file__).parent / "Mobilenetv3large_paddy_disease_detection_architecture_3_fine_tuned.keras"
model = load_model(model_path)


labels_path = base_path / "labels.txt"

# load class names
with open(labels_path, 'r') as f:
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