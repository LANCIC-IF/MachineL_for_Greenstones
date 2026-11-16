import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


#plotting libraries
import plotly.express as px
import plotly.graph_objects as go

#------------------------------functions------------------------------

def PCA (X: np.ndarray):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)
    
    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort the eigenvalues and eigenvectors in ascending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices] * -1 #(-1) is used to invert the signs of the eigenvectors that numpy returns
    
    return sorted_eigenvalues, sorted_eigenvectors
    #return eigenvalues, eigenvectors
    
    
def proyecting_data (X: np.ndarray, eigenvectors: np.ndarray, n_components: int) -> np.ndarray:
    # Select the top n_components eigenvectors
    top_eigenvectors = eigenvectors[:, :n_components]
    # Project the data onto the new feature space
    X_projected = np.dot(X, top_eigenvectors)
    
    return X_projected
    

def generar_pca_3d(
    labels: list[tuple], 
    eigenvalues: np.ndarray, 
    projected_data: np.ndarray, 
    n_components: int = 3
) -> go.Figure:
    
    # Validar que el número de componentes sea compatible con un gráfico 3D
    if n_components != 3:
        raise ValueError("n_components debe ser exactamente 3 para proyectar un gráfico 3D.")
        
    # Calcular el porcentaje de varianza explicada por los 3 primeros componentes
    total_variance = np.sum(eigenvalues)
    var_explained = [(eig / total_variance) * 100 for eig in eigenvalues[:3]]
    
    # Desempaquetar la lista de tuplas en dos listas separadas
    etiqueta_principal = [lbl[0] for lbl in labels]
    etiqueta_secundaria = [lbl[1] for lbl in labels]
    
    # Crear un DataFrame para facilitar la graficación y categorización
    df = pd.DataFrame({
        'PC1': projected_data[:, 0],
        'PC2': projected_data[:, 1],
        'PC3': projected_data[:, 2],
        'Etiqueta Principal': etiqueta_principal,
        'Etiqueta Secundaria': etiqueta_secundaria
    })
    
    # Construir el gráfico 3D
    fig = px.scatter_3d(
        df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Etiqueta Principal',      # Genera la leyenda lateral por color
        symbol='Etiqueta Secundaria',    # Genera la leyenda lateral por forma/símbolo
        title='Proyección PCA 3D',
        labels={
            'PC1': f'PC1 ({var_explained[0]:.1f}%)',
            'PC2': f'PC2 ({var_explained[1]:.1f}%)',
            'PC3': f'PC3 ({var_explained[2]:.1f}%)'
        }
    )
    
    # Personalizar la estética de los puntos y la información al acercar el mouse (hover)
    fig.update_traces(
        marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')),
        customdata=df[['Etiqueta Principal', 'Etiqueta Secundaria']],
        hovertemplate=(
            "<b>Principal:</b> %{customdata[0]}<br>"
            "<b>Secundaria:</b> %{customdata[1]}<br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}<br>"
            "PC3: %{z:.3f}<extra></extra>" # <extra></extra> oculta el recuadro secundario redundante
        )
    )
    
    # Ajustar proporciones de la cámara para que la rotación sea fluida y el espacio equilibrado
    fig.update_layout(
        scene=dict(
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig