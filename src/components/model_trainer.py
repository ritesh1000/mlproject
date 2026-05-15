import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.Model_Trainer_Config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training and Testing data:")
            x_train, y_train, x_test, y_test= (
                                                train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                              )
            
            models={
                        "Linear Regression": LinearRegression(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(),
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor()
                   }
            
            model_report: dict=evaluate_model(x_train,y_train, x_test,y_test,models)

            ## To get best model score from Dict
            best_model_score= max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            
            logging.info("Best model found on both training and testing data")

            save_object(
                file_path=self.Model_Trainer_Config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            logging.info("best model found is :",best_model)
            logging.info("R2 Score of best model is : ",r2_square)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
        


