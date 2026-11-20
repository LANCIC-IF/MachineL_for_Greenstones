import numpy as np
import pandas as pd 

#ML libraries
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#plot libraries 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

#---------------------------------PCA and Kmeans----------------------------------------

def whole_data_Kmeans(DB: np.ndarray, n_clusters: int, random_state: int = 14, test_size: float = 0.2):
    
    X_train, X_test = train_test_split(DB, test_size= test_size, random_state= random_state)
    
    kmeans = KMeans(n_clusters = n_clusters, random_state = random_state, n_init= "auto").fit(X_train)
    predictions = kmeans.predict(X_test)
    centroids = kmeans.cluster_centers_
    
    return predictions, centroids

def generar_pca_3d_completo(
    labels: list[tuple], 
    eigenvalues: np.ndarray, 
    projected_data: np.ndarray, 
    predicciones: np.ndarray,
    centroids: np.ndarray,
    random_state: int = 14
) -> go.Figure:
    n_samples = len(projected_data)
    if len(labels) != n_samples:
        raise ValueError("Debe haber una etiqueta por cada punto proyectado.")
    if len(predicciones) > n_samples:
        raise ValueError("No puede haber más predicciones que puntos proyectados.")

    total_variance = np.sum(eigenvalues)
    var_explained = [(eig / total_variance) * 100 for eig in eigenvalues[:3]]

    if len(predicciones) == 0:
        test_indices = np.array([], dtype=int)
    elif len(predicciones) == n_samples:
        test_indices = np.arange(n_samples)
    else:
        _, test_indices = train_test_split(
            np.arange(n_samples),
            test_size=len(predicciones),
            random_state=random_state
        )

    split_labels = np.full(n_samples, "Entrenamiento", dtype=object)
    split_labels[test_indices] = "Prueba"
    cluster_labels = np.full(n_samples, "No evaluado", dtype=object)
    cluster_labels[test_indices] = np.asarray(predicciones).astype(str)

    df = pd.DataFrame({
        'PC1': projected_data[:, 0],
        'PC2': projected_data[:, 1],
        'PC3': projected_data[:, 2],
        'Pieza': [lbl[0] for lbl in labels],
        'Lado': [lbl[1] for lbl in labels],
        'Seccion': [lbl[2] for lbl in labels],
        'Punto': [lbl[3] for lbl in labels],
        'Conjunto': split_labels,
        'Cluster K-Means': cluster_labels
    })

    piece_values = df['Pieza'].drop_duplicates().tolist()
    palette = px.colors.qualitative.Plotly
    color_map = {
        piece: palette[index % len(palette)]
        for index, piece in enumerate(piece_values)
    }

    fig = px.scatter_3d(
        df,
        x='PC1', y='PC2', z='PC3',
        color='Pieza',
        symbol='Conjunto',
        symbol_map={'Entrenamiento': 'circle', 'Prueba': 'diamond'},
        color_discrete_map=color_map,
        custom_data=['Pieza', 'Lado', 'Seccion', 'Punto', 'Cluster K-Means'],
        title=f'Proyección PCA 3D - Espacio: {df["Lado"].iloc[0].upper()}',
        labels={
            'PC1': f'PC1 ({var_explained[0]:.1f}%)',
            'PC2': f'PC2 ({var_explained[1]:.1f}%)',
            'PC3': f'PC3 ({var_explained[2]:.1f}%)'
        }
    )

    fig.update_traces(
        marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate=(
            "<b>Pieza:</b> %{customdata[0]}<br>"
            "<b>Lado:</b> %{customdata[1]}<br>"
            "<b>Sección:</b> %{customdata[2]} | <b>Punto:</b> %{customdata[3]}<br>"
            "<b>Cluster K-Means:</b> %{customdata[4]}<br>"
            "---------------------<br>"
            "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
        )
    )

    if len(test_indices):
        x_radius = max(np.ptp(projected_data[:, 0]) * 0.012, 1e-6)
        y_radius = max(np.ptp(projected_data[:, 1]) * 0.012, 1e-6)
        test_data = df.iloc[test_indices]

        for piece in piece_values:
            piece_test_data = test_data[test_data['Pieza'] == piece]
            if piece_test_data.empty:
                continue

            star_x, star_y, star_z = [], [], []
            for _, point in piece_test_data.iterrows():
                for vertex in range(11):
                    angle = -np.pi / 2 + vertex * np.pi / 5
                    radius = 1 if vertex % 2 == 0 else 0.42
                    star_x.append(point['PC1'] + np.cos(angle) * x_radius * radius)
                    star_y.append(point['PC2'] + np.sin(angle) * y_radius * radius)
                    star_z.append(point['PC3'])
                star_x.append(None)
                star_y.append(None)
                star_z.append(None)

            fig.add_trace(go.Scatter3d(
                x=star_x,
                y=star_y,
                z=star_z,
                mode='lines',
                line=dict(color=color_map[piece], width=3),
                name=f'Prueba - {piece}',
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        mode='markers',
        marker=dict(size=14, symbol='diamond', color='black', line=dict(width=2, color='white')),
        name='Centroides',
        hovertemplate="<b>Centroide</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(scene=dict(aspectmode='cube'), margin=dict(l=0, r=0, b=0, t=40))

    return fig


def confusion_matrix_of_clas ():
    return 1 

#group fold by K-menas and PCA
def evaluate_kfold_kmeans(labels: list[tuple], data: np.ndarray, n_splits: int = 3, random_state: int = 4) -> list[dict]:
    """
    Realiza la validación cruzada agrupando por pieza de jade y retorna 
    una lista de diccionarios con toda la información necesaria para graficar cada fold.
    """
    groups = np.array([lbl[0] for lbl in labels])
    gkf = GroupKFold(n_splits=n_splits)
    resultados_folds = []

    for fold_idx, (train_index, test_index) in enumerate(gkf.split(data, groups=groups)):
        X_train, X_test = data[train_index], data[test_index]
        
        # PCA
        pca = PCA(n_components=3)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # K-Means (ajustado a 3 clústeres como mencionaste)
        kmeans = KMeans(n_clusters=3, random_state = random_state)
        kmeans.fit(X_train_pca)
        
        predicciones = kmeans.predict(X_test_pca)
        
        # Extraer las etiquetas exactas de los datos que quedaron en validación
        etiquetas_test = [labels[i] for i in test_index]
        
        # Empaquetamos la salida
        fold_data = {
            'fold_numero': fold_idx + 1,
            'projected_data': X_test_pca,
            'eigenvalues': pca.explained_variance_,
            'centroids': kmeans.cluster_centers_,
            'labels': etiquetas_test,
            'predicciones_kmeans': predicciones
        }
        resultados_folds.append(fold_data)
        
    return resultados_folds


#plotting PCA and Kmeans results
def generar_pca_3d_desde_fold(fold_data: dict) -> go.Figure:
    """
    Recibe el diccionario generado por evaluate_kfold_kmeans y renderiza el espacio 3D.
    """
    projected_data = fold_data['projected_data']
    eigenvalues = fold_data['eigenvalues']
    labels = fold_data['labels']
    centroids = fold_data['centroids']
    predicciones = fold_data['predicciones_kmeans']
    fold_num = fold_data['fold_numero']

    # Cálculo de varianza
    total_variance = np.sum(eigenvalues)
    var_explained = [(eig / total_variance) * 100 for eig in eigenvalues[:3]]
    
    # Desempaquetado para Plotly
    etiqueta_principal = [lbl[0] for lbl in labels]
    etiqueta_secundaria = [lbl[1] for lbl in labels]
    
    df = pd.DataFrame({
        'PC1': projected_data[:, 0],
        'PC2': projected_data[:, 1],
        'PC3': projected_data[:, 2],
        'Etiqueta Principal': etiqueta_principal,
        'Etiqueta Secundaria': etiqueta_secundaria,
        'Cluster Predicho': predicciones
    })
    
    fig = px.scatter_3d(
        df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Etiqueta Principal',      
        symbol='Etiqueta Secundaria',    
        title=f'Proyección PCA 3D - Fold {fold_num}',
        labels={
            'PC1': f'PC1 ({var_explained[0]:.1f}%)',
            'PC2': f'PC2 ({var_explained[1]:.1f}%)',
            'PC3': f'PC3 ({var_explained[2]:.1f}%)'
        }
    )
    
    fig.update_traces(
        marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')),
        customdata=df[['Etiqueta Principal', 'Etiqueta Secundaria', 'Cluster Predicho']],
        hovertemplate=(
            "<b>Principal:</b> %{customdata[0]}<br>"
            "<b>Secundaria:</b> %{customdata[1]}<br>"
            "<b>K-Means Cluster:</b> %{customdata[2]}<br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}<br>"
            "PC3: %{z:.3f}<extra></extra>"
        )
    )

    # Inyección de Centroides
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(size=12, symbol='diamond', color='black', line=dict(width=2, color='white')),
        name='Centroides (Entrenamiento)',
        hovertemplate=(
            "<b>Centroide K-Means</b><br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}<br>"
            "PC3: %{z:.3f}<extra></extra>"
        )
    ))
    
    fig.update_layout(
        scene=dict(aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig