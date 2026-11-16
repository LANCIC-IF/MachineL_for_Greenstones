import pytest
import numpy as np

from Basic_Functions.PCA import PCA, proyecting_data

#-----------------------manual numerical case -----------------------

example_data = np.array([[1, -1],[0, 1],[-1, 0]])

#-----------------manual method 
#eigenvectors returned from higher to lower eigenvalue and with the same sign as the manual calculation
manual_eigenvectors = np.array([[1, 1],[-1, 1]]) * (1 / np.sqrt(2))

#eigenvalues from manual calculation
manual_eigenvalues = np.array([3/2, 1/2])

#manual prpyected data onto the new feature space using the manual eigenvectors
manual_projected_data = np.array([[2, 0],[-1, 1],[-1, -1]]) * (1 / np.sqrt(2))

#-----------------numpy method
#numpy returns from lower to higher eigenvalue and inverted signs from the manual calculation
numpy_eigenvectors = np.array([[-1, -1], [-1, 1]]) * (1 / np.sqrt(2)) 

#numpy returns from lower to higher eigenvalue
numpy_eigenvalues = np.array([1/2,3/2]) 

numpy_projected_data = np.array([[0, -np.sqrt(2)], [1, 0],[-1, np.sqrt(2)]])


#-------------------------------tests------------------------------

def test_PCA():
    assert np.allclose(PCA(example_data)[0], manual_eigenvalues, rtol=1e-20) and np.allclose(PCA(example_data)[1], manual_eigenvectors, rtol=1e-20)


def test_PCA_projected_data():
    assert np.allclose(proyecting_data(example_data, PCA(example_data)[1], 2), manual_projected_data, rtol=1e-20)


""" Note: I have to test how does other libraries like sklearn and scipy calculate the PCA, 
because i want to be sure that the results are reproducible with other libraries  """