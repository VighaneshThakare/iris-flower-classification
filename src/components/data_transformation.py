import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

from src.exception import CustomException

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "Species"

            X_train = train_df.drop(columns=[target_column, "Id"])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column, "Id"])
            y_test = test_df[target_column]

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Encode labels
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            # Save preprocessor
            with open(self.preprocessor_obj_file_path, "wb") as f:
                pickle.dump((scaler, label_encoder), f)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)