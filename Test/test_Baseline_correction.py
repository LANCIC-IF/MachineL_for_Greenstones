import pytest 

#importing functions
from Basic_Functions.reading_functions_fromDB import load_DB_from_JSON, get_spectrum_from_DB

example_DB = r"C:\Users\liatc\OneDrive\Documentos\GreenStone_Yaxchilan\Database\Yaxchilan_Alpha1_septiembre_2019_tumba_1_Reflectance_DB.json"

def test_load_DB_from_JSON(DB: dict = load_DB_from_JSON(example_DB)):
    assert isinstance(DB, dict), "The loaded DB should be a dictionary."
    assert len(DB) > 0, "The loaded DB should not be empty."
    
