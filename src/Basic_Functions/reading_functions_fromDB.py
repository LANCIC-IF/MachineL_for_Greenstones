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

#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    DB = load_DB_from_JSON("path_to_your_json_file.json")