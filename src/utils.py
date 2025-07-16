import sys
import os
import pandas as pd
import numpy as np
import pickle
import dill

from src.logger import logging
from src.exception import CustomException

from pymongo import MongoClient
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

########################################

def export_collection_as_dataframe(collection_name,db_name):
    try:
        mongo_client = MongoClient("mongodb+srv://pragya:Tech26@cluster0.icm43.mongodb.net/?retryWrites=true&w=majority&tls=true&appName=Cluster0")
        collection = mongo_client[db_name][collection_name]

        # Print debug information
        print("Connection string :",mongo_client)
        print("Database name:",db_name)
        print("Connection mame:",collection_name)

        num_samples = collection.count_documents({})  #count the number of documents in the collection
        if num_samples == 0:
            raise ValueError("The collection is empty.Please ensure there are samples in the collection.")
        
        df = pd.DataFrame(list(collection.find()))

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"],axis=1)
        df.replace({"na": np.nan}, inplace=True)

        return df
    except Exception as e:
        raise CustomException(e,sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            # Train model
            # model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)


            # Predict testing data
            y_test_pred = model.predict(x_test)

            # Get R2 score for train and test data
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.info("Error occured during model training")
        raise CustomException(e,sys)

# def load_object(file_path):
#     try:
#         with open(file_path,'rb') as file_obj:
#             return dill.load(file_obj)
#     except Exception as e:
#         logging.info('Exception Occured in load_object function utils')
#         raise CustomException(e,sys)





