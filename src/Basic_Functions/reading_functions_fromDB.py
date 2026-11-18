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

    #reference_name = [key.split('_')[0].lower() for key in DB.keys()]
    #front_or_Back = [key.split('_')[1].lower() for key in DB.keys()] 
    #section = [key.split('_')[2].split(".")[0].lower() for key in DB.keys()]
    #number_of_spectrum = [key.split('_')[2].split(".")[1].lower() for key in DB.keys()]

    parsed_keys = [(key.split('_')[0].lower(), key.split('_')[1].lower(), key.split('_')[2].split(".")[0].lower(), key.split('_')[2].split(".")[1].lower()) for key in DB.keys()]

    #return list(zip(reference_name, front_or_Back, section, number_of_spectrum))
    return parsed_keys
    

#used to match labels and reflectance data by position 
def retriving_labels_and_data_from_DB(DB: dict) -> tuple:
    labels = retriving_labels_from_keys(DB)
    data = np.array(list(DB.values()))
    return labels, data



def removing_specific_datafrom_label (DB: dict, label_to_remove: str) -> dict:
    try:

        clean_DB = {key: label for key, label in DB.items() if label_to_remove.lower() not in key.lower()}
        return clean_DB

    except Exception as e:
        print(f"Error occurred while removing specific data from the database: {e}")
        return DB  # Return the original DB if an error occurs
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    "theres nothing here"