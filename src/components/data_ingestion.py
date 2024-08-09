import os 
import pandas as pd 
from src.logger import logging
from src.excepation import CustomExcepation 
import sys 
from sklearn.model_selection  import train_test_split
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("Artifacts", "train.csv")
    test_data_path:str=os.path.join("Artifacts", "test.csv")
    raw_data_path:str=os.path.join("Artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_save=DataIngestionConfig()
        
        
        
        
    def initiate_data_ingestion(self):
        logging.info("Start the Data Ingestion Method ")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read The data")
            
            
            os.makedirs(os.path.dirname(self.data_ingestion_save.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_save.train_data_path,index=False,header=True)
            logging.info("Train Test Split inititate")
            
            
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_save.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info(" Data Ingestion of the data is completed")
            
            
            return(
                self.data_ingestion_save.train_data_path,
                self.data_ingestion_save.test_data_path
    
                
            )
            
           
            
            
        except Exception as e :
            raise CustomExcepation(e,sys)