import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer #Handling missing values
from sklearn.preprocessing import StandardScaler #Handling feature scaling
from sklearn.preprocessing import OrdinalEncoder #Ordinal encoding - ranking the categorical features
from sklearn.pipeline import Pipeline #pipeline - to combine multiple steps
from sklearn.compose import ColumnTransformer #To combine pipelines
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            
            #Define which column should be ordinal-encoded and which should be scaled
            logging.info("Defining categorical and numerical columns")
            categorical_cols = ['cut', 'color', 'clarity'] #objects
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z'] #numerical

            #Define the custom ranking for each ordinal variable
            logging.info("Defining ordinal variables used for ranking")
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
                ]
            )

            ##Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())    
                ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_cols),
            ('cat_pipeline', cat_pipeline, categorical_cols)    
            ])

            logging.info("Pipeline created")

            return preprocessor
            
        except Exception as e:
            logging.info("Exception occured at Data Transformation stage")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and Test data")
            logging.info(f"Train Head : \n {train_df.head().to_string()}")
            logging.info(f"Test Head : \n {test_df.head().to_string()}")
            
            logging.info("Obtaining Preprocessing object")
            preprocessingObj = self.get_data_transformation_object()

            #Feature Engineering or Preprocessing
            target_column = 'price'
            drop_column = [target_column, 'id'] 

            input_feature_train_df = train_df.drop(columns = drop_column, axis=1) #Train Features
            target_feature_train_df = train_df[target_column] #Train Target

            input_feature_test_df = test_df.drop(columns = drop_column, axis=1) #Test Features
            target_feature_test_df = test_df[target_column] #Test Target

            #Transformation
            logging.info("Applying preprocessing object on training and testing datasets")
            input_feature_train_arr = preprocessingObj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessingObj.transform(input_feature_test_df)

            #Concatinating Features and Target into numpy array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocess_obj_file_path, 
                obj=preprocessingObj
            )

            logging.info("Preprocessor pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured at Initiate Data Transformation stage")
            raise CustomException(e, sys)

