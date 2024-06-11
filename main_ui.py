import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from cure import cures

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/latestmodel_pdd.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class_index]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence_score

def update_cure_info(trace, points, selector):
    # Function to update cure information based on selected bar
    selected_class = points.point_inds[0]
    selected_class_name = class_names[selected_class]
    if selected_class_name in cures:
        st.subheader("Cure Information:")
        st.write(cures[selected_class_name])
    else:
        st.subheader("Cure Information:")
        st.write("Cure information not available.")

st.title('🌱 Plant Disease Identifier 🌿')
st.write("This classification model can be used to identify the diseases of a variety of plants like apple, blueberry, cherry, corn, grape, orange, pepper, potato, strawberry, tomato")

uploaded_image = st.file_uploader("Upload an image... (Supported formats: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.subheader("Uploaded Image:")
    image = Image.open(uploaded_image)
    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        st.write("")
        if st.button('Predict'):  # Changed button text here
            prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
            st.subheader("Prediction:")
            st.success(f'Prediction: {prediction}')
            st.write("")
            st.subheader("Confidence Score:")
            st.success(f'Confidence: {confidence:.2f}')

            class_names = list(class_indices.values())
            confidence_scores = model.predict(load_and_preprocess_image(uploaded_image))[0]
            fig = go.Figure(data=[go.Bar(x=class_names, y=confidence_scores)])
            fig.update_layout(title="Confidence Scores for Predicted Classes",
                              xaxis_title="Class Name",
                              yaxis_title="Confidence Score",
                              xaxis_tickangle=-45,
                              hovermode="closest",  # Show hover information closest to the cursor
                              clickmode="event+select"  # Enable click selection on bars
                              )

            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.2f}",
                marker_color='rgb(158,202,225)',  # Customize bar color
                marker_line_color='rgb(8,48,107)',  # Customize bar outline color
                marker_line_width=1.5  # Customize bar outline width
            )

            fig.data[0].on_click(update_cure_info)

            st.plotly_chart(fig, use_container_width=True)
            if prediction in cures:
                cure_info = cures[prediction]
            else:
                cure_info = "Cure information not available."

            st.subheader("Cure Information:")
            st.write(cure_info)