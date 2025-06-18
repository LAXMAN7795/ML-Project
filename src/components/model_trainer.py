import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score, accuracy_score
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered the model trainer component")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'LinearRegression': LinearRegression(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_name = max(model_report, key=lambda k: model_report[k]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model: {r2_square}")
            return r2_square, best_model_name, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
