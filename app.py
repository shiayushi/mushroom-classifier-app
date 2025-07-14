import streamlit as st
import pandas as pd
import joblib

model = joblib.load("mushroom_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Input from user
cap_shape = st.selectbox("Cap Shape", encoders['cap-shape'].classes_)
# ... other inputs ...

# Convert input to encoded form
input_data = {
    'cap-shape': encoders['cap-shape'].transform([cap_shape])[0],
    # ... other encoded inputs ...
}
df = pd.DataFrame([input_data])

# Predict and show result
prediction = model.predict(df)[0]
st.write("🍽️ Edible" if prediction == 'e' else "☠️ Poisonous")
