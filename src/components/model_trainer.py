import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.h5")

    def build_model(self, input_dim):
        model = models.Sequential([
            layers.Dense(16, activation='relu', input_shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training started")

            model = self.build_model(X_train.shape[1])

            model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=8,
                validation_data=(X_test, y_test),
                verbose=0
            )

            y_pred = np.argmax(model.predict(X_test), axis=1)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Model accuracy: {accuracy}")

            model.save(self.model_path)

            logging.info("Model saved successfully")

        except Exception as e:
            raise CustomException(e, sys)