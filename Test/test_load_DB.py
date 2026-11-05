import pytest 
import numpy as np 
#importing internal functions
from Basic_Functions.reading_functions_fromDB import load_DB_from_JSON, get_spectrum_from_DB


#--------------Testing DB's -------------------------------------------------------

example_DB = r"C:\Users\liatc\OneDrive\Documentos\GreenStone_Yaxchilan\Database\Yaxchilan_Alpha1_septiembre_2019_tumba_1_Reflectance_DB.json"
empty_DB = r"C:\Users\liatc\OneDrive\Documentos\GreenStone_Yaxchilan\Test\empty_json.json"

#--------------Test for load DB correctly-----------------

def test_load_DB_from_JSON_isnotempty(DB: dict = load_DB_from_JSON(example_DB)):
    assert len(DB) > 0, "The database should not be empty."


def test_load_DB_from_JSON_isempty(DB: dict = load_DB_from_JSON(empty_DB)):
    assert len(DB) == 0, "The database should be empty."


def test_get_spectrum_from_DB_existing_key(DB: dict = load_DB_from_JSON(example_DB)):
    # Assuming 'existing_filename' is a key in the example_DB
    existing_filename = next(iter(DB.keys()))  # Get the first key from the DB
    spectrum = get_spectrum_from_DB(DB, existing_filename)
    assert spectrum.size > 0 and type(spectrum) == np.ndarray      


def test_get_spectrum_from_DB_non_existing_key(DB: dict = load_DB_from_JSON(example_DB)):
    non_existing_filename = "muestra3_anaquel4.0"
    spectrum = get_spectrum_from_DB(DB, non_existing_filename)
    assert spectrum.size == 0 and type(spectrum) == np.ndarray
    

def test_retriving_keys_from_DB(DB: dict = load_DB_from_JSON(example_DB)):
    keys = list(DB.keys())
    assert len(keys) > 0 and type(keys) == list, "The keys should be a non-empty list."
    

def test_retriving_keys_from_DB_empty(DB: dict = load_DB_from_JSON(empty_DB)):
    keys = list(DB.keys())
    assert len(keys) == 0 and type(keys) == list, "The keys should be an empty list."
    

