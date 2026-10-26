import numpy as np
from pybaselines import Baseline, utils
import matplotlib.pyplot as plt


def wavenumber(complete_spectra: np.ndarray, start_wavenumber: float, end_wavenumber: float) -> np.ndarray:
    
    n_points = complete_spectra.shape[0]
    return np.linspace(start_wavenumber, end_wavenumber, n_points)
                       
                       
def wavenumber_to_index(wavenumber: np.ndarray, goal_wavenumber: float) -> np.ndarray:
    
    step_of_spectra = wavenumber[1] - wavenumber[0]
    index = int(round(goal_wavenumber - wavenumber[0]) / step_of_spectra)
    return index

def slicing_all_BD(spectra_DB: dict, start: int, end: int) -> dict:
    sliced_DB = {}
    for filename, spectra in spectra_DB.items():
        sliced_DB[filename] = spectra[start:end + 1]
    return sliced_DB

def baseline_correction(spectra: np.ndarray, wavenumber: np.ndarray) -> np.ndarray:
    baseline_fitter = Baseline(x_data = wavenumber)
    baseline, params = baseline_fitter.modpoly(spectra, poly_order=1)
    
    return spectra - baseline

def batch_baseline_correction(spectra_DB: dict, wavenumber: np.ndarray) -> dict:
    corrected_baseline_DB = {}
    for filename, spectra in spectra_DB.items():
        corrected_spectra = baseline_correction(spectra, wavenumber)
        corrected_baseline_DB[filename] = corrected_spectra
    return corrected_baseline_DB

def plot_baseline_correction(spectra: np.ndarray, corrected_spectra: np.ndarray, wavenumber: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber, spectra, label='Original Spectra', color='blue')
    plt.plot(wavenumber, corrected_spectra, label='Corrected Spectra', color='green')
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.title('Baseline Correction')
    plt.legend()
    plt.show()
    
    
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    
    #defining simulated signals
    x = np.linspace(0, 1000, 1000)
    
    simulated_spectra1 = (utils.gaussian(x, 4, 50, 10) + utils.gaussian(x, 4, 70, 5) + 2*np.exp((-x/400)))  # Adding a small Gaussian noise
    simulated_spectra2 = (utils.gaussian(x, 4, 60, 8) + utils.gaussian(x, 4, 80, 6) + 2*np.exp((-x/400)))  # Adding a small Gaussian noise
    
    spectra_DB = {
        "file1": simulated_spectra1,
        "file2": simulated_spectra2
    }
    
    #defining slicing wavenumber range 
    min_value = 0
    max_value = 1000
    goal_value = 500
    
    sliced_DB = slicing_all_BD(spectra_DB, min_value, goal_value)
    print("length of sliced_DB:", len(sliced_DB["file1"]))  # Should print 501
    
    
    #making baseline correction
    
    sliced_wavenumber = wavenumber(sliced_DB["file1"], min_value, goal_value)
    corrected_baseline_DB = batch_baseline_correction(sliced_DB, sliced_wavenumber)
    
    plot_baseline_correction(sliced_DB["file1"], corrected_baseline_DB["file1"], sliced_wavenumber)    