import os
<<<<<<< HEAD
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Successfully saved object at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)
=======
import dill

def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)

def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)
>>>>>>> d743ec0794036bbb46ae63f1cece06729a8a3539
