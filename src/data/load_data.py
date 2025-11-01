import os 
import sys
import gzip
import json
import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.exception import CustomException 
from src.utils.utils import download_file
from src.utils.config_parser import load_config

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class LOAD_DATA:
    
    def __init__(self, raw_data_yaml:str = 'configs\config.yaml'):
        try:
            self.config= load_config(raw_data_yaml)
            self.logger = logger

            paths=self.config.get('paths', {})
            load_cfg = self.config.get('data_ingestion', {})
            self.save_dir= load_cfg.get('save_dir')
            
            
            pass
        except Exception as e:
            raise CustomException(e)
    def load_data(self, file_name):
        try:
            file_path = os.path.join(self.raw_data_path, file_name)
            pass

        except Exception as e:
            raise CustomException(e)