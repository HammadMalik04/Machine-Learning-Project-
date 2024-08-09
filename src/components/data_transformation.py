import os 
import sys
from src.logger import logging
from src.excepation import CustomExcepation
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    Preprocessor_obj_file_path=os.path.join("Artifacts","preprocessor.pkl") 
    
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_save=DataTransformation()
        
        
        
    def get_data_transformation(self):
        logging.info("Start Data Transformation")
        
        try:
            numerical_columns=["Writing_score", " reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]
            num_pipeline=Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )
            
            cate_pipeline= Pipeline(
                steps=[
                    ("impute",SimpleImputer(stragtegy="most_frequent")),
                    ("one Hot Encoder ", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            
            
            
            logging.info(f"categrical_columns: {categorical_columns}")
            logging.info(f"numerical_columns: {numerical_columns}")
            
            
            
            perprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)
                    ("cate_pipeline",cate_pipeline,categorical_columns)
                    
                ]
            )
            
            
            return perprocessor
        except Exception as e :
            raise CustomExcepation (e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read Train and Test data Completed ")
            logging.info("Obtaining Preprocessor object")
            
            
            preprocessing_obj=self.get_data_transformation()
            
           
            target_column_name="math_score"
            numerical_columns=["reading_score","writing_score"]
            
            
            input_feature_train_data=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_data =train_df[target_column_name] 
            
            
            input_feature_test_data=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_data=test_df[target_column_name]
            
            logging.info(f"Applying preprocessing  object on training dataframe and testing dataframe.")
            
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_data)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_data)
                
            ]
            
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_data)
            ]
            
            logging.info(f"save preprocessing object")
            
            
            save_object(
                file_path=self.data_transformation_save.preprocessing_obj_file_path
                obj=preprocessing_obj
            )
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_save.preprocessing_obj_file_path,
                )
                     
            
            
            
        except Exception as e :
            raise CustomExcepation (e,sys)
            
