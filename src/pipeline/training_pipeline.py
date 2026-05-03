from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(
            X_train, X_test, y_train, y_test
        )

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()