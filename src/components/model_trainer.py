import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from src.exception import CustomException


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
            model = self.build_model(X_train.shape[1])

            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=8,
                validation_data=(X_test, y_test),
                verbose=1
            )

            # DL Predictions
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"DL Model Accuracy: {accuracy}")

            os.makedirs("artifacts", exist_ok=True)

            # -------------------------------
            # Confusion Matrix
            # -------------------------------
            cm = confusion_matrix(y_test, y_pred)

            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig("artifacts/confusion_matrix.png")
            plt.close()

            # -------------------------------
            # Feature Importance (ML model)
            # -------------------------------
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)

            importances = rf.feature_importances_

            feature_names = [
                "SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"
            ]

            plt.figure()
            sns.barplot(x=importances, y=feature_names)
            plt.title("Feature Importance (Random Forest)")
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.savefig("artifacts/feature_importance.png")
            plt.close()

            # Save DL model
            model.save(self.model_path)

        except Exception as e:
            raise CustomException(e, sys)