import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# class CustomModel:
#     def __init__(self, preprocessing_object, trained_model_object):
#         self.preprocessing_object = preprocessing_object

#         self.trained_model_object = trained_model_object

#     def predict(self, X):
#         transformed_feature = self.preprocessing_object.transform(X)

#         return self.trained_model_object.predict(transformed_feature)

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"

#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            x_train, y_train,x_test,y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(algorithm="SAMME")
            }

            param = {"Random Forest":{
                'criterion':['gini'],
                'min_samples_leaf':[1,5,10,15],
                'n_estimators':[10,50,100]
            },
            "Decision Tree":{
                'criterion':['gini','entropy','log_loss'],
                'splitter':["best","random"],
                'min_samples_split':[2,4,6,8,10],
                'min_samples_leaf':[1,10,20,40]
            },
            "Gradient Boosting":{
                'loss':["log_loss","exponential"],
                'learning_rate':[0.1,0.5,0.6],
                'criterion':['friedman_mse','squared_error']
            },
            "K-Neighbors Classifier":{
               'n_neighbors':[5,10,15],
                # 'weights':['uniform','distance'],
                # 'algorithm':['auto','ball_tree','kd_tree','brute'],
               'leaf_size':[30,40] 
            },
            "XGBClassifier":{
                'n_estimators':[2,4,6,8,10],
                'max_depth':[2,4,6],
                'learning_rate':[0.5,1]       
            },
            "AdaBoost Classifier":{
                'learning_rate':[.1,.01,0.5,.001],
                'n_estimators':[8,16,32,64,128,256]
            }
            }

            logging.info(f"Extracting model config file path")

            model_report:dict = evaluate_models(x_train=x_train, y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=param)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found",sys)

            logging.info(f"Best found model on both training and testing dataset")

            # preprocessing_obj = load_object(file_path=preprocessor_path)

            # custom_model = CustomModel(
            #     preprocessing_object=preprocessing_obj,
            #     trained_model_object=best_model,
            # )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_file_path}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test,predicted)
            return accuracy
        except Exception as e:
            logging.info("Exception occured at model traning")
            raise CustomException(e,sys)
            

            