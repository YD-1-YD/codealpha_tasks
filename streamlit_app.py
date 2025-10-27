import streamlit as st
import pandas as pd
import joblib

# Load the trained model
#it will load model
# it will load the model from the specified path that it should be in same folder as this file
# Make sure to adjust the path if necessary
model = joblib.load(r"C:\Users\ydmal\OneDrive\Desktop\IRIS\iris_best_model (2).joblib")

# App title
st.title("IRIS FLOWER CLASSIFICATION")
st.write("This app predicts the species of an Iris flower based on sepal and petal measurements.")


st.sidebar.header("Enter flower measurements (in cm)")

sepal_length = st.sidebar.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.1, step=0.1, format="%.1f")
sepal_width  = st.sidebar.number_input("Sepal Width", min_value=2.0, max_value=4.5, value=3.5, step=0.1, format="%.1f")
petal_length = st.sidebar.number_input("Petal Length", min_value=1.0, max_value=7.0, value=1.4, step=0.1, format="%.1f")
petal_width  = st.sidebar.number_input("Petal Width", min_value=0.1, max_value=2.5, value=0.2, step=0.1, format="%.1f")


sample = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(sample)


st.subheader("Prediction")
st.write(f"🌼 The model predicts this is a **{prediction[0]}** flower.")

# Shows user input while changing with sliders.
st.subheader("Your Input")
st.write(pd.DataFrame(
    [[f"{sepal_length} cm", f"{sepal_width} cm", f"{petal_length} cm", f"{petal_width} cm"]],
    columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
))
