import numpy as np 
import matplotlib.pyplot as plt

#ML libraries
import sklearn as sk

#plotting libraries
import plotly 




def PCA (X):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)
    
    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices] * -1
    
    return sorted_eigenvalues, sorted_eigenvectors
    #return eigenvalues, eigenvectors
    
    
def proyecting_data (X, eigenvectors, n_components):
    # Select the top n_components eigenvectors
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # Project the data onto the new feature space
    X_projected = np.dot(X, top_eigenvectors)
    
    return X_projected
    

def movile_PCAplot ():
    return 1