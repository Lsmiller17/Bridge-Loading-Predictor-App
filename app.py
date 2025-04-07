
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load model and preprocessor
model = tf.keras.models.load_model("tf_bridge_model.h5")
preprocessor = joblib.load("preprocessor.pkl")

# Streamlit app layout
st.title("Bridge Max Load Predictor")

span = st.number_input("Span (ft)", min_value=1)
deck_width = st.number_input("Deck Width (ft)", min_value=1)
age = st.number_input("Age (Years)", min_value=0)
lanes = st.number_input("Number of Lanes", min_value=1)
material = st.selectbox("Material", ["Steel", "Concrete", "Composite"])
condition = st.slider("Condition Rating", 1, 5)

if st.button("Predict Max Load (Tons)"):
    input_df = pd.DataFrame([{
        "Span_ft": span,
        "Deck_Width_ft": deck_width,
        "Age_Years": age,
        "Num_Lanes": lanes,
        "Material": material,
        "Condition_Rating": condition
    }])
    X_input = preprocessor.transform(input_df)
    prediction = model.predict(X_input)
    st.success(f"Predicted Max Load: {prediction[0][0]:.2f} tons")
