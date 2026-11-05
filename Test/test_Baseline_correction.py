import pytest
import numpy as np

from Basic_Functions.Baseline_correction import (
    wavenumber,
    wavenumber_to_index,
    wavenumber_to_index_reflectance_array,
    slicing_all_BD_position,
    slicing_all_BD_wavenumber,
    baseline_correction,
    batch_baseline_correction
)

#----------------------------Testing baseline corrcection functions---------------------------------

#test to find the index of a specific wavenumber value when you have explicitly defined wavenumber array
@pytest.mark.parametrize("wavenumber, goal_wavenumber, expected_index", [(np.linspace(400, 4000, 1764), 400, 0), 
                                                                          (np.linspace(400, 4000, 1764), 4000, 1763), 
                                                                          (np.linspace(400, 4000, 1764), 2200, 882)])
def test_wavenumber_to_index(wavenumber, goal_wavenumber, expected_index):
    assert wavenumber_to_index(wavenumber, goal_wavenumber) == expected_index


#test to find the index of a specific wavenumber value when you only have reflectance array
@pytest.mark.parametrize("reflectance, left_wn, right_wn, goal_wavenumber, expected_index", [(np.linspace(24, 76, 1764), 400, 4000, 400, 0), 
                                                                                          (np.linspace(0, 1200, 1764), 400, 4000, 4000, 1763), 
                                                                                          (np.linspace(122, 440, 1764), 400, 4000, 2200, 882)])
def test_wavenumber_to_index_reflectance_array(reflectance, left_wn, right_wn, goal_wavenumber, expected_index):
    assert wavenumber_to_index_reflectance_array(reflectance, left_wn, right_wn, goal_wavenumber) == expected_index


#test to find the index of a specific wavenumber value when you only have reflectance array with inverted wavenumber range
@pytest.mark.parametrize("reflectance, left_wn, right_wn, goal_wavenumber, expected_index", [(np.linspace(24, 76, 1764), 4000, 400, 400, 1763), 
                                                                                          (np.linspace(0, 1200, 1764), 4000, 400, 4000, 0), 
                                                                                          (np.linspace(122, 440, 1764), 4000, 400, 2200, 882)])
def test_wavenumber_to_index_reflectance_array_inverted_wavenmber(reflectance, left_wn, right_wn, goal_wavenumber, expected_index):
    assert wavenumber_to_index_reflectance_array(reflectance, left_wn, right_wn, goal_wavenumber) == expected_index


#test slice all DB when u know the exact position of the slice
@pytest.mark.parametrize("spectra_DB, start, end, expected_sliced_DB", [
    ({"file1": np.array([1, 2, 3, 2, 1]), "file2": np.array([1,3,5,3,1])}, 1, 4, {"file1": np.array([2, 3, 2, 1]), "file2": np.array([3, 5, 3, 1])})])
def test_slicing_all_BD_only_position(spectra_DB, start, end, expected_sliced_DB):
    sliced_DB = slicing_all_BD_position(spectra_DB, start, end)
    assert np.array_equal(sliced_DB["file1"], expected_sliced_DB["file1"]) and np.array_equal(sliced_DB["file2"], expected_sliced_DB["file2"])


#test slice all DB when u only know the wavenumber range of the slice with normal and inverted wavenumber range
@pytest.mark.parametrize("spectra_DB, left_wn, right_wn, startslice_wavenumber, endslice_wavenumber, expected_sliced_DB", [
    ({"file1": np.array([1, 2, 3, 2, 1]), "file2": np.array([1,3,5,3,1])},400, 4000, 400, 1200, {"file1": np.array([1,2]), "file2": np.array([1,3])}),
    ({"file1": np.array([1, 2, 3, 2, 1]), "file2": np.array([1,3,5,3,1])},4000, 400, 3100, 400, {"file1": np.array([2,3,2,1]), "file2": np.array([3,5,3,1])})])
def test_slicing_all_BD_only_wavenumber(spectra_DB, left_wn, right_wn, startslice_wavenumber, endslice_wavenumber, expected_sliced_DB):
    sliced_DB = slicing_all_BD_wavenumber(spectra_DB, left_wn, right_wn, startslice_wavenumber, endslice_wavenumber)
    assert np.array_equal(sliced_DB["file1"], expected_sliced_DB["file1"]) and np.array_equal(sliced_DB["file2"], expected_sliced_DB["file2"])


#test baselinecorrection of a single spectra
@pytest.mark.parametrize("spectra, wn, poly_order, corrected_spectra", [
    (np.array([1, 2, 3, 2, 3, 4, 5, 4, 3, 2]), np.linspace(400, 4000, 10), 1, np.array([0, 1, 2, 1, 2, 3, 4, 3, 2, 0]))])    
def test_baseline_correction(spectra, wn, poly_order, corrected_spectra):
    assert np.array_equal(baseline_correction(spectra, wn, poly_order), corrected_spectra)


#always fix the spectra to cero baseline 
@pytest.mark.parametrize("spectra_DB, wn, poly_order, expected_corrected_baseline_DB", [
    ({"file1": np.array([1, 2, 3, 2, 3, 4, 5, 4, 3, 2]), "file2": np.array([2, 3, 4, 3, 4, 5, 6, 5, 4, 3])}, 
     np.linspace(400, 4000, 10), 1,
     {"file1": np.array([0, 1, 2, 1, 2, 3, 4, 3, 2, 0]), "file2": np.array([0, 1, 2, 1, 2, 3, 4, 3, 2, 0])})])
def test_batch_baseline_correction(spectra_DB, wn, poly_order, expected_corrected_baseline_DB):
    corrected_baseline_DB = batch_baseline_correction(spectra_DB, wn, poly_order)
    for filename in spectra_DB.keys():
        assert np.array_equal(corrected_baseline_DB[filename], expected_corrected_baseline_DB[filename])
    