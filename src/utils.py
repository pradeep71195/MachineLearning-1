#Common methods/functions
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occured at Save Object function")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report= {}
        model_list = []
        r2_list = []

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            #Make Predictions
            y_pred = model.predict(X_test)

            mae, rmse, r2_square = get_evaluation_score(y_test, y_pred)

            print(list(models.keys())[i])
            model_list.append(list(models.keys())[i])
            r2_list.append(r2_square)
            test_score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = test_score

            print("Model Training Performance")
            print("RMSE: ", rmse)
            print("MAE: ", mae)
            print("R2 Square: ", r2_square*100)

            

            print("="*30)
            print("\n")

        return report    
    except Exception as e:
        logging.info("Exception occured at evaluate_model function")
        raise CustomException(e, sys)

def get_evaluation_score(test, predicted):
    mae = mean_absolute_error(test, predicted)
    mse = mean_squared_error(test, predicted)
    rmse = np.sqrt(mean_squared_error(test, predicted))
    r2_square = r2_score(test, predicted)

    return mae, rmse, r2_square