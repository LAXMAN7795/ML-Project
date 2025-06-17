import os
import sys
from src.exception import CustomException # Importing CustomException from src.exception
from src.logger import logging # Importing logging from src.logger

from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

@dataclass # Using dataclass to define a configuration class directly
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')  # Reading the dataset from a CSV file
            logging.info("Read the dataset as a pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # Create directories if they do not exist
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # Saving the raw data to a CSV file
            logging.info("Saved the raw data to CSV")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Split the dataset into train and test sets")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # Saving the train set to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # Saving the test set to a CSV file
            logging.info("Saved train and test sets to CSV files")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()