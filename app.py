import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Ladda modellen och testdata
mlp = joblib.load("mlp_model.joblib")
X_test = joblib.load("X_test.joblib")
y_test = joblib.load("y_test.joblib")

# Funktion för att förutsäga siffra
def predict_digit(image):
    img = image.resize((28, 28)).convert('L')
    img_array = np.array(img) / 255.0
    img_flat = img_array.reshape(1, -1)
    prediction = mlp.predict(img_flat)[0]
    return prediction

# Streamlit-app
st.title("MNIST Klassificerare")
st.write("Ladda upp en bild på en siffra (0-9) för att få en förutsägelse.")

# Ladda upp bild
uploaded_file = st.file_uploader("Ladda upp bild...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uppladdad bild", width=150)

    # Knapp för att förutsäga
    if st.button("Förutsäg"):
        prediction = predict_digit(image)
        st.write(f"Förutsägelse: **{prediction}**")

# Visa testnoggrannhet
if st.checkbox("Visa testnoggrannhet"):
    test_acc = mlp.score(X_test, y_test)
    st.write(f"Testnoggrannhet: {test_acc:.4f}")
