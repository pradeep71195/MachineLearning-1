import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

## Initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

## Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method started")
        try:
            #Read input dataset
            df = pd.read_csv(os.path.join("./notebooks/data","gemstone.csv"))
            logging.info("Dataset read!!")
            #Create folder if not exist already
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            #Write df to csv in raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("train test split")
            train_set, test_set = train_test_split(df, test_size=0.3)

            #Write Train and Test to CSv files respectively
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)


