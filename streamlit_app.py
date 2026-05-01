import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os
import json

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Iris Classifier", layout="centered")

USER_FILE = "users.json"

# -------------------------------
# Load / Save Users
# -------------------------------
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

users = load_users()

# -------------------------------
# Session State
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "inputs" not in st.session_state:
    st.session_state.inputs = {"sl": 5.5, "sw": 3.0, "pl": 4.0, "pw": 1.2}

if "result" not in st.session_state:
    st.session_state.result = None

# -------------------------------
# Auth Functions
# -------------------------------
def login(username, password):
    if username in users and users[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.rerun()
    else:
        st.error("Invalid credentials")

def register(username, password):
    if username in users:
        st.error("User already exists")
    elif username == "" or password == "":
        st.warning("Fields cannot be empty")
    else:
        users[username] = password
        save_users(users)
        st.success("Registration successful! Please login.")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# -------------------------------
# LOGIN / REGISTER
# -------------------------------
if not st.session_state.logged_in:

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            login(username, password)

    with tab2:
        st.subheader("Register")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            register(new_user, new_pass)

    st.stop()

# -------------------------------
# Load Model
# -------------------------------
model = tf.keras.models.load_model("artifacts/model.h5")
scaler, label_encoder = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
df = pd.read_csv("notebook/data/iris.csv")

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.title("🌸 Iris Flower Classification")
st.caption(f"Welcome, {st.session_state.username}")

if st.button("Logout"):
    logout()

# -------------------------------
# Prediction Function
# -------------------------------
def run_prediction(sl, sw, pl, pw):
    data = pd.DataFrame({
        "SepalLengthCm": [sl],
        "SepalWidthCm": [sw],
        "PetalLengthCm": [pl],
        "PetalWidthCm": [pw],
    })

    scaled = scaler.transform(data)
    probs = model.predict(scaled)
    pred = np.argmax(probs, axis=1)

    label = label_encoder.inverse_transform(pred)[0]
    confidence = round(np.max(probs) * 100, 2)

    class_probs = {
        label_encoder.inverse_transform([i])[0]: round(p * 100, 2)
        for i, p in enumerate(probs[0])
    }

    return label, confidence, class_probs

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Insights", "ℹ️ About"])

# ===============================
# 🔮 Prediction
# ===============================
with tab1:

    st.subheader("Enter Flower Measurements")

    col1, col2 = st.columns(2)

    with col1:
        sl = st.slider("Sepal Length", 4.0, 8.0, st.session_state.inputs["sl"])
        sw = st.slider("Sepal Width", 2.0, 4.5, st.session_state.inputs["sw"])

    with col2:
        pl = st.slider("Petal Length", 1.0, 7.0, st.session_state.inputs["pl"])
        pw = st.slider("Petal Width", 0.1, 2.5, st.session_state.inputs["pw"])

    b1, b2, b3 = st.columns(3)

    # Predict
    with b1:
        if st.button("Predict", use_container_width=True):
            st.session_state.result = run_prediction(sl, sw, pl, pw)

    # Sample
    with b2:
        if st.button("Sample Values", use_container_width=True):
            sample = df.sample(1).iloc[0]
            st.session_state.inputs = {
                "sl": float(sample["SepalLengthCm"]),
                "sw": float(sample["SepalWidthCm"]),
                "pl": float(sample["PetalLengthCm"]),
                "pw": float(sample["PetalWidthCm"])
            }
            st.rerun()

    # Reset
    with b3:
        if st.button("Reset", use_container_width=True):
            st.session_state.inputs = {"sl": 5.5, "sw": 3.0, "pl": 4.0, "pw": 1.2}
            st.session_state.result = None
            st.rerun()

    # Result
    if st.session_state.result:

        label, confidence, class_probs = st.session_state.result

        st.markdown("### Prediction Result")

        c1, c2 = st.columns(2)
        c1.metric("Predicted Class", label)
        c2.metric("Confidence", f"{confidence:.2f}%")

        st.progress(int(confidence))

        # ✅ FIXED FLOWER IMAGES
        st.markdown("### Predicted Flower")

        if "setosa" in label.lower():
            st.image("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris-images/iris-setosa.jpg")

        elif "versicolor" in label.lower():
            st.image("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris-images/iris-versicolor.jpg")

        else:
            st.image("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris-images/iris-virginica.jpg")

        # Probabilities
        prob_df = pd.DataFrame(list(class_probs.items()), columns=["Class", "Probability"])
        st.bar_chart(prob_df.set_index("Class"))

        # Comparison
        avg = df.drop(columns=["Id", "Species"]).mean()
        compare_df = pd.DataFrame({
            "Your Input": [sl, sw, pl, pw],
            "Average": avg.values
        }, index=["SL", "SW", "PL", "PW"])

        st.bar_chart(compare_df)

        st.info("Prediction mainly depends on petal features.")

# ===============================
# 📊 Insights (EXPANDERS)
# ===============================
with tab2:

    st.subheader("Model Insights")

    with st.expander("📊 Show Confusion Matrix"):
        if os.path.exists("artifacts/confusion_matrix.png"):
            st.image("artifacts/confusion_matrix.png")

    with st.expander("📈 Show Feature Importance"):
        if os.path.exists("artifacts/feature_importance.png"):
            st.image("artifacts/feature_importance.png")

# ===============================
# ℹ️ About
# ===============================
with tab3:

    st.header("📘 Project Overview")

    st.write("""
This project uses Deep Learning (Neural Networks) to classify Iris flowers 
based on sepal and petal measurements.

### Model:
- Neural Network (Keras)
- ReLU activation
- Softmax output layer

### Features:
- Real-time prediction  
- Confidence visualization  
- Sample data generator  
- Reset functionality  
- Model insights visualization  
- User authentication system  

### Dataset:
- 150 samples  
- 3 classes  

### Tech Stack:
- TensorFlow  
- Scikit-learn  
- Pandas  
- Streamlit  
""")

    st.subheader("👨‍💻 Team")
    st.write("Vighanesh Thakare and Team")