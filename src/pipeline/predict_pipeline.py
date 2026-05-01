import sys
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.h5")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            model = tf.keras.models.load_model(self.model_path)

            with open(self.preprocessor_path, "rb") as f:
                scaler, label_encoder = pickle.load(f)

            scaled_data = scaler.transform(features)

            probs = model.predict(scaled_data)
            pred = np.argmax(probs, axis=1)

            label = label_encoder.inverse_transform(pred)[0]
            confidence = round(np.max(probs) * 100, 2)

            return label, confidence

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame({
                "SepalLengthCm": [self.sepal_length],
                "SepalWidthCm": [self.sepal_width],
                "PetalLengthCm": [self.petal_length],
                "PetalWidthCm": [self.petal_width],
            })

        except Exception as e:
            raise CustomException(e, sys)