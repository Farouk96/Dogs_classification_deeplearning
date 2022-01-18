# import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import keras
from PIL import Image
import cv2
import numpy as np

# Displaying in-head
st.title("Dog Breed Classification - MobileNetV2")

st.text(" Téléchargez une image de chien pour obtenir sa race. Seulement 120 races disponibles.")

# Importing labels name
my_content = open("dogs_name.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()

def image_classifier(img, weights_file):
  """Function which classifies an image. 
  Inputs: Image to classify & Model to load
  Output : Prediction with the greater probability
  """
  model = keras.models.load_model(weights_file)
  image = img_to_array(img)
  image = cv2.resize(image,(224,224))
  image = image.reshape(1,224,224,3)
  predictions = model.predict(image)
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions)
  return dogs_list[predictions]

# Displaying uploader
uploaded_file = st.file_uploader("Téléchargez votre image...", type="jpg")

# Loop ending with prediction
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption=' Image téléchargée.', use_column_width=True)
  st.write("")
  st.write(" Classification...")
  label = image_classifier(img, 'my_model.h5')
  st.write(label)
