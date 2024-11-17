import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from bs4 import BeautifulSoup

model = tf.keras.applications.MobileNetV2(weights='imagenet')

calorie_data = pd.read_csv('calorie_data.csv')  

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def predict_fruit(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)

    fruit_name = decoded_predictions[0][0][1]
    print(f"Predicted Fruit: {fruit_name}")

    cal = fetch_calories(fruit_name)
    print(f"calories : {cal}")

    
predict_fruit('apple.png')
