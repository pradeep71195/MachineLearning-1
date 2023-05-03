import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    dt_obj = DataTransformation()
    train_arr, test_arr,preprocessor_path = dt_obj.initiate_data_transformation(train_data, test_data)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
    