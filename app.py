import streamlit as st
import tensorflow as tf
import PIL as Image
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


img = cv2.imread("1836.jpg")
img = cv2.resize(img, (224, 224))
img = np.array(img)
expand_img = np.expand_dims(img, axis=0)
prepro_img = preprocess_input(expand_img)
res = model.predict(prepro_img).flatten()
normalized = res / norm(res)


neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distance,indices = neighbors.kneighbors([normalized])

print(indices)

for file in indices[0][0:5]:
    imgName = cv2.imread(file_name[file])
    cv2.imshow("Image", (imgName))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()