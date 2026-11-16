from cProfile import label
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_DB_from_JSON(input_file: str) -> dict:
    try: 
        with open(input_file, 'r') as f:
            data = json.load(f)
        # Convert lists back to numpy arrays
        DB = {key: np.array(value) for key, value in data.items()}
        return DB
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in file: {input_file}")
        print(f"Error: {e.msg}")
        return {}

def get_spectrum_from_DB(DB: dict, filename: str) -> np.ndarray:
    try:
        return DB[filename]
    except KeyError:
        print(f"Filename '{filename}' not found in the database.")
        return np.array([])  # Return an empty array if the filename is not found

    
def retriving_keys_from_DB(DB: dict) -> list:
    try:
        return list(DB.keys())
    except Exception as e:
        print(f"Error occurred while retrieving keys from the database: {e}")
        return []


def retriving_labels_from_keys(DB: dict) -> list:
    try:
        first_label = [key.split('_')[0].lower() for key in DB.keys()]
        second_label = [key.split('_')[1].lower() for key in DB.keys()] 
        
        return list(zip(first_label, second_label))
    except Exception as e:
        print(f"Error occurred while retrieving labels from keys: {e}, check the format of the keys in the database.")
        return []


def removing_specific_datafrom_label (DB: dict, label_to_remove: str) -> dict:
    try:

        clean_DB = {key: label for key, label in DB.items() if label_to_remove.lower() not in key.lower()}
        return clean_DB

    except Exception as e:
        print(f"Error occurred while removing specific data from the database: {e}")
        return DB  # Return the original DB if an error occurs
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    DB = load_DB_from_JSON("path_to_your_json_file.json")