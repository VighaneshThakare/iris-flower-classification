from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import sys

from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)


def load_objects():
    model = tf.keras.models.load_model("artifacts/model.h5", compile=False)
    scaler, label_encoder = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
    df = pd.read_csv("notebook/data/iris.csv")
    return model, scaler, label_encoder, df


def predict(sl, sw, pl, pw):
    model, scaler, label_encoder, _ = load_objects()

    data = pd.DataFrame({
        "SepalLengthCm": [float(sl)],
        "SepalWidthCm":  [float(sw)],
        "PetalLengthCm": [float(pl)],
        "PetalWidthCm":  [float(pw)]
    })

    scaled = scaler.transform(data)
    probs = model.predict(scaled)

    pred = np.argmax(probs, axis=1)
    label = label_encoder.inverse_transform(pred)[0]
    confidence = round(float(np.max(probs)) * 100, 2)

    class_probs = {
        label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(probs[0])
    }

    return label, confidence, class_probs


# HOME PAGE
@app.route("/")
def home():
    return render_template("home.html")


# CLASSIFIER PAGE
@app.route("/classifier")
def classifier():
    return render_template("index.html")


# PREDICT (JSON API)
@app.route("/predict", methods=["POST"])
def make_prediction():
    try:
        logging.info("Prediction request received")

        data = request.get_json()

        if not data:
            logging.warning("No input data received")
            return jsonify({"error": "No input data provided"}), 400

        sl = data.get("sl")
        sw = data.get("sw")
        pl = data.get("pl")
        pw = data.get("pw")

        if None in [sl, sw, pl, pw]:
            logging.warning("Missing input values")
            return jsonify({"error": "Missing input values"}), 400

        logging.info(f"Input values - SL: {sl}, SW: {sw}, PL: {pl}, PW: {pw}")

        label, confidence, class_probs = predict(sl, sw, pl, pw)

        logging.info(f"Prediction: {label} with confidence {confidence}%")

        return jsonify({
            "label": label,
            "confidence": confidence,
            "probs": class_probs
        })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)


# SAMPLE DATA (JSON)
@app.route("/sample")
def sample():
    _, _, _, df = load_objects()
    row = df.sample(1).iloc[0]

    return jsonify({
        "sl": round(float(row["SepalLengthCm"]), 1),
        "sw": round(float(row["SepalWidthCm"]), 1),
        "pl": round(float(row["PetalLengthCm"]), 1),
        "pw": round(float(row["PetalWidthCm"]), 1)
    })


if __name__ == "__main__":
    print("Starting Flask App...")
    app.run(debug=True)