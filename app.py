import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🕵️",
    layout="wide"
)

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource
def load_models():
    deepfake_model = tf.keras.models.load_model(
        "models/deepfake_model.keras"
    )

    gender_model = tf.keras.models.load_model(
        "models/gender_model.keras"
    )

    age_model = tf.keras.models.load_model(
        "models/age_model.keras"
    )

    return deepfake_model, gender_model, age_model


deepfake_model, gender_model, age_model = load_models()

# ----------------------------
# Class Labels
# ----------------------------
AGE_CLASSES = [
    "18-25",
    "26-35",
    "36-50",
    "50+"
]

# gender mapping
# {'female': 0, 'male': 1}

# ----------------------------
# UI
# ----------------------------
st.title("🕵️ Deepfake Detection & Face Analysis System")

st.markdown("""
Upload a facial image and predict:

- Deepfake Status
- Gender
- Age Group
""")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------
if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    resized_img = img.resize((224, 224))

    img_array = np.array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------------
    # Deepfake Prediction
    # ------------------------
    deepfake_pred = deepfake_model.predict(
        img_array,
        verbose=0
    )[0][0]

    # ------------------------
    # Gender Prediction
    # ------------------------
    gender_pred = gender_model.predict(
        img_array,
        verbose=0
    )[0][0]
    
    # ------------------------
    # Age Prediction
    # ------------------------
    age_pred = age_model.predict(
        img_array,
        verbose=0
    )[0]

    age_idx = np.argmax(age_pred)
    age_group = AGE_CLASSES[age_idx]
    age_conf = age_pred[age_idx] * 100

    st.subheader("Prediction Results")

    # Deepfake Result
    if deepfake_pred > 0.5:
        st.success(
            f"✅ REAL Image ({deepfake_pred*100:.2f}% confidence)"
        )
    else:
        st.error(
            f"🚨 FAKE Image ({(1-deepfake_pred)*100:.2f}% confidence)"
        )

    # Gender Result
    if gender_pred > 0.5:
        gender = "Male"
        gender_conf = gender_pred * 100
    else:
        gender = "Female"
        gender_conf = (1 - gender_pred) * 100

    st.info(
        f"👤 Gender: {gender} ({gender_conf:.2f}% confidence)"
    )

    # Age Result
    st.warning(
        f"🎂 Age Group: {age_group} ({age_conf:.2f}% confidence)"
    )

    st.write("Raw Gender Score:", float(gender_pred))
    st.write("Raw Age Scores:", age_pred.tolist())
    st.caption(
    "Deepfake Detection, Gender Classification and Age Group Prediction"
)
    