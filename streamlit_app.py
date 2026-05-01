import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os

# -------------------------------
# Load artifacts (DL FIX)
# -------------------------------
model = tf.keras.models.load_model("artifacts/model.h5")
scaler, label_encoder = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

# Dataset for random sample
df = pd.read_csv("notebook/data/iris.csv")

st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("🌸 Iris Flower Classification")
st.caption("Predict species using sepal and petal measurements")

# -------------------------------
# Helpers
# -------------------------------
def safe_float(value):
    try:
        return float(value)
    except:
        return None

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
# Session state
# -------------------------------
if "inputs" not in st.session_state:
    st.session_state.inputs = {"sl": "", "sw": "", "pl": "", "pw": ""}

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -------------------------------
# Tabs (RESTORED UI)
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Insights", "ℹ️ About"])

# ===============================
# 🔮 Prediction Tab
# ===============================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        sl = st.text_input("Sepal Length", value=st.session_state.inputs["sl"])
        sw = st.text_input("Sepal Width", value=st.session_state.inputs["sw"])

    with col2:
        pl = st.text_input("Petal Length", value=st.session_state.inputs["pl"])
        pw = st.text_input("Petal Width", value=st.session_state.inputs["pw"])

    c1, c2, c3 = st.columns(3)

    # Predict
    with c1:
        if st.button("Predict", use_container_width=True):
            sl_f = safe_float(sl)
            sw_f = safe_float(sw)
            pl_f = safe_float(pl)
            pw_f = safe_float(pw)

            if None in [sl_f, sw_f, pl_f, pw_f]:
                st.error("Please enter valid numeric values.")
            else:
                label, confidence, class_probs = run_prediction(sl_f, sw_f, pl_f, pw_f)
                st.session_state.last_result = (label, confidence, class_probs)
                st.session_state.inputs = {"sl": sl, "sw": sw, "pl": pl, "pw": pw}

    # Random Sample (same feature retained)
    with c2:
        if st.button("Random Sample", use_container_width=True):
            sample = df.sample(1).iloc[0]

            st.session_state.inputs = {
                "sl": str(sample["SepalLengthCm"]),
                "sw": str(sample["SepalWidthCm"]),
                "pl": str(sample["PetalLengthCm"]),
                "pw": str(sample["PetalWidthCm"]),
            }

            st.rerun()

    # Reset
    with c3:
        if st.button("Reset", use_container_width=True):
            st.session_state.inputs = {"sl": "", "sw": "", "pl": "", "pw": ""}
            st.session_state.last_result = None
            st.rerun()

    # Result display
    if st.session_state.last_result:
        label, confidence, class_probs = st.session_state.last_result

        st.markdown("### Result")
        m1, m2 = st.columns(2)
        m1.metric("Predicted Class", label)
        m2.metric("Confidence", f"{confidence}%")

        st.markdown("### Class Probabilities (%)")
        prob_df = pd.DataFrame(
            list(class_probs.items()), columns=["Class", "Probability"]
        )
        st.bar_chart(prob_df.set_index("Class"))

# ===============================
# 📊 Insights Tab
# ===============================
with tab2:

    st.markdown("### Model Visualizations")

    with st.expander("Show Confusion Matrix"):
        if os.path.exists("artifacts/confusion_matrix.png"):
            st.image("artifacts/confusion_matrix.png")
        else:
            st.warning("Run training to generate this.")

    with st.expander("Show Feature Importance"):
        if os.path.exists("artifacts/feature_importance.png"):
            st.image("artifacts/feature_importance.png")
        else:
            st.warning("Run training to generate this.")

# ===============================
# ℹ️ About Tab
# ===============================
with tab3:

    st.header("📘 Project Overview")

    st.markdown("""
This project implements a **hybrid Machine Learning and Deep Learning system** 
to classify Iris flower species based on morphological features.

The system is designed to demonstrate:
- Predictive modeling using deep learning
- Model evaluation and interpretability
- Real-time deployment using Streamlit
""")

    st.divider()

    st.subheader("🧠 Model Details")

    st.markdown("""
**Primary Model (Deep Learning):**
- Feedforward Neural Network (Keras Sequential)
- Dense layers with ReLU activation
- Softmax output layer for multi-class classification
- Loss Function: Sparse Categorical Crossentropy
- Optimizer: Adam

**Supporting Model (Machine Learning):**
- Random Forest Classifier
- Used for feature importance (interpretability)
""")

    st.divider()

    st.subheader("📊 Dataset Information")

    st.markdown("""
**Dataset:** Iris Dataset  

**Features:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Target Classes:**
- Setosa
- Versicolor
- Virginica
""")

    st.divider()

    st.subheader("⚙️ System Workflow")

    st.markdown("""
1. User inputs flower measurements  
2. Data is preprocessed using StandardScaler  
3. Neural network predicts class probabilities  
4. Highest probability → final prediction  
5. Confidence score is displayed  
6. Model insights (confusion matrix & feature importance) are available  
""")

    st.divider()

    st.subheader("📈 Model Evaluation")

    st.markdown("""
- **Accuracy:** Measures overall correctness  
- **Confusion Matrix:** Shows class-wise performance  
- **Feature Importance:** Identifies most influential features  
""")

    st.divider()

    st.subheader("💡 Key Features")

    st.markdown("""
- Real-time prediction  
- Confidence score display  
- Random sample generator  
- Interactive visualization  
- Hybrid ML + DL architecture  
""")

    st.divider()

    st.subheader("🎯 Justification of Approach")

    st.markdown("""
- Deep learning is used for predictive modeling  
- Random Forest is used for interpretability since neural networks lack transparency  
- This hybrid approach balances **performance and explainability**  
""")

    st.divider()

    st.subheader("🚀 Conclusion")

    st.markdown("""
The system demonstrates how deep learning models can be deployed in real-world 
applications while maintaining interpretability using traditional machine learning techniques.
""")