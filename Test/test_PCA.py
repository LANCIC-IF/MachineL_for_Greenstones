import pytest
import numpy as np

from Basic_Functions.PCA import PCA 

#-----------------------manual numerical case -----------------------

example_data = np.array([[1, -1],
                            [0, 1],
                            [-1, 0]])


#eigenvectors returned from lower to higher eigenvalue and inverted signs from the manual calculation
manual_eigenvectors = np.array([[1, 1],
                                [-1, 1]]) * (1 / np.sqrt(2))

#eigenvalues from manual calculation
manual_eigenvalues = np.array([3/2, 1/2])

#manual prpyected data onto the new feature space using the manual eigenvectors
manual_projected_data = np.array([[2, 0],
                            [-1, 1],
                            [-1, -1]]) * (1 / np.sqrt(2))


numpy_eigenvectors = np.array([[-1, -1],
                                [-1, 1]]) * (1 / np.sqrt(2)) #numpy returns from lower to higher eigenvalue and inverted signs from the manual calculation

numpy_eigenvalues = np.array([1/2,3/2]) #numpy returns from lower to higher eigenvalue

numpy_projected_data = np.array([[0, -np.sqrt(2)],
                            [1, 0],
                            [-1, np.sqrt(2)]])




def test_PCA():
    assert np.allclose(PCA(example_data)[0], manual_eigenvalues, rtol=1e-20) and np.allclose(PCA(example_data)[1], manual_eigenvectors, rtol=1e-20)