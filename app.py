import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import shutil

from flask import Flask, request, render_template

app = Flask(__name__)

# -------------------------------
# Load Model & Preprocessor
# -------------------------------
model = tf.keras.models.load_model("artifacts/model.h5")

with open("artifacts/preprocessor.pkl", "rb") as f:
    scaler, label_encoder = pickle.load(f)

df = pd.read_csv("notebook/data/iris.csv")

# -------------------------------
# Move graphs to static (auto)
# -------------------------------
os.makedirs("static", exist_ok=True)

if os.path.exists("artifacts/confusion_matrix.png"):
    shutil.copy("artifacts/confusion_matrix.png", "static/confusion_matrix.png")

if os.path.exists("artifacts/feature_importance.png"):
    shutil.copy("artifacts/feature_importance.png", "static/feature_importance.png")

# -------------------------------
# Helpers
# -------------------------------
def safe_float(value):
    try:
        return float(value)
    except:
        return None


def predict_values(sl, sw, pl, pw):
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

    return label, confidence


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    sl = safe_float(request.form.get("sepal_length"))
    sw = safe_float(request.form.get("sepal_width"))
    pl = safe_float(request.form.get("petal_length"))
    pw = safe_float(request.form.get("petal_width"))

    if None in [sl, sw, pl, pw]:
        return render_template("index.html", error="Invalid input")

    label, confidence = predict_values(sl, sw, pl, pw)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        sl=sl, sw=sw, pl=pl, pw=pw
    )


@app.route("/random")
def random_sample():
    sample = df.sample(1).iloc[0]

    return render_template(
        "index.html",
        sl=sample["SepalLengthCm"],
        sw=sample["SepalWidthCm"],
        pl=sample["PetalLengthCm"],
        pw=sample["PetalWidthCm"]
    )


if __name__ == "__main__":
    app.run(debug=True)