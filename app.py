from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
from src.ML_Project.components.data_ingestion import DataIngestion
from src.ML_Project.components.data_ingestion import DataIngestionConfig
from src.ML_Project.components.data_transformation import DataTransformationConfig, DataTransformation
from src.ML_Project.components.model_trainer import ModelTrainer,ModelTrainerConfig


import sys



if __name__=="__main__":
    logging.info("the execution has started")

    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.intiate_data_transformation(train_data_path,test_data_path)

        ## Model Taining

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_tranier(train_array=train_arr,test_array=test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)