import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'  # Path to the model
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)  # Load the model
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)  # Scale the features
            predictions = model.predict(data_scaled)  # Make predictions
            return predictions
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, # Define the parameters for the custom data
                 gender, 
                 race_ethnicity, 
                 parental_level_of_education, 
                 lunch, 
                 test_preparation_course, 
                 reading_score, 
                 writing_score):
        self.gender = gender # Gender of the student
        self.race_ethnicity = race_ethnicity # Race/Ethnicity of the student
        self.parental_level_of_education = parental_level_of_education # Parental level of education
        self.lunch = lunch # Lunch type
        self.test_preparation_course = test_preparation_course # Test preparation course status
        self.reading_score = reading_score # Reading score
        self.writing_score = writing_score # Writing score

    def get_data_as_dataframe(self):
        """
        This method converts the custom data into a pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [float(self.reading_score)],
                "writing_score": [float(self.writing_score)]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
