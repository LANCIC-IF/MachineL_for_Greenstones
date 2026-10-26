import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_DB_from_JSON(input_file: str) -> dict:
    with open(input_file, 'r') as f:
        data = json.load(f)
    # Convert lists back to numpy arrays
    DB = {key: np.array(value) for key, value in data.items()}
    return DB

def get_spectrum_from_DB(DB: dict, filename: str) -> np.ndarray:
    return DB.get(filename, "Filename not found in DB")


    
def retriving_keys_from_DB(DB: dict) -> list:
    return list(DB.keys())

#----------------------------------------------------------------------------------------------


if __name__ == "__main__":
    DB = load_DB_from_JSON("path_to_your_json_file.json")