# main.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Input
from tensorflow.keras.models import Model
import cv2
import pickle
from sklearn.neighbors import NearestNeighbors

# Load the feature list and file names
feature_list = np.array(pickle.load(open('feature_vector.pkl', 'rb')))
file_name = pickle.load(open('filenames.pkl', 'rb'))

# Define the base model with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)

# Add Input layer with shape specification
inputs = Input(shape=(224, 224, 3))

# Pass the inputs through the base model
x = base_model(inputs, training=False)

# Add GlobalMaxPooling2D layer
outputs = GlobalMaxPooling2D()(x)

# Define the final model
model = Model(inputs, outputs)

# Print model summary
model.summary()

# Define feature extraction function
def extract_features(img, model):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img).flatten()
    normalized_features = features / norm(features)
    return normalized_features

# Streamlit UI
st.title("Fashion Recommendation System")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract features from the image
    features = extract_features(img, model)

    # Find nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    # Display the recommended images in 5 columns
    st.write("similar products")
    cols = st.columns(5)
    for i, idx in enumerate(indices[0]):
        recommended_img_path = file_name[idx]
        recommended_img = Image.open(recommended_img_path)
        with cols[i]:
            st.image(recommended_img, caption=f"Image {i + 1}", use_column_width=True)
