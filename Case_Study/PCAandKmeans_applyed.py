import numpy as np 
import matplotlib.pyplot as plt 

#ML libraries
import sklearn 

#internal lib
from Basic_Functions.reading_functions_fromDB import *
from Basic_Functions.Baseline_correction import *
from Basic_Functions.PCA import *  
from Basic_Functions.Kmeans import *

#-------------------------------------- Data --------------------------------------------
references = r"C:\Users\liatc\OneDrive\Documentos\GreenStone_Yaxchilan\Database\JADE_mapping_reflectance.json"

#---------------------------------cleaning and separating data --------------------------
complete_DB = load_DB_from_JSON(references)
removing_motagua = removing_specific_datafrom_label(complete_DB, "Motagua") #Because it doesn't have the same polished surface

#-------------------------------Slicing and Baseline correction--------------------------
left_wavenumber = 4000
right_wavenumber = 400
left_slice_wavenumber = 1350
right_slice_wavenumber = 400

slicing_without_motagua = slicing_all_BD_wavenumber(removing_motagua, left_wavenumber, right_wavenumber, left_slice_wavenumber, right_slice_wavenumber)

wavenumber = np.linspace(left_slice_wavenumber, right_slice_wavenumber, len(next(iter(slicing_without_motagua.values()))))
baseline_without_motagua = batch_baseline_correction(slicing_without_motagua, wavenumber ,1)

#--------------------------------Dividing front and back --------------------------------
front_data = removing_specific_datafrom_label(baseline_without_motagua, "reverso") #len: 192
back_data = removing_specific_datafrom_label( baseline_without_motagua, "frente")#len: 191

#----------------------------------Retriving labels --------------------------------------
front_labels, front_data = retriving_labels_and_data_from_DB(front_data)
back_labels, back_data  = retriving_labels_and_data_from_DB(back_data)

print("front data shape: ", front_data.shape)
print("back data shape: ", back_data.shape)

#------------------------------Plotting baseline corrected data ------------------------

def plot_spectra(data, title):
	fig, ax = plt.subplots(figsize=(12, 6))
	colors = plt.cm.turbo(np.linspace(0, 1, len(data)))

	for spectrum, color in zip(data, colors):
		ax.plot(wavenumber, spectrum, color=color, alpha=0.35, linewidth=0.8)

	ax.set_title(title)
	ax.set_xlabel("Wavenumber (cm$^{-1}$)")
	ax.set_ylabel("Reflectance")
	ax.grid(True, alpha=0.2)
	fig.tight_layout()
	return fig, ax


front_fig, front_ax = plot_spectra(front_data, "Front spectra")
back_fig, back_ax = plot_spectra(back_data, "Back spectra")
plt.show()

#---------------------------------PCA and Kmeans----------------------------------------

#Whole Dataset PCA and Kmeans----

front_sorted_eigenvalues, front_sorted_eigenvectors = manual_PCA(front_data)
back_sorted_eigenvalues, back_sorted_eigenvectors = manual_PCA(back_data)

front_proyected_data = proyecting_data(front_data, front_sorted_eigenvectors, 3)
back_proyected_data = proyecting_data(back_data, back_sorted_eigenvectors, 3)

front_predictions, front_centroids = whole_data_Kmeans(front_proyected_data, 3, 14, 0.2)
back_predictions, back_centroids = whole_data_Kmeans(back_proyected_data, 3, 14, 0.2)

figura = generar_pca_3d_completo(
    labels=front_labels,
    eigenvalues=front_sorted_eigenvalues,
    projected_data=front_proyected_data,
    predicciones=front_predictions,
    centroids=front_centroids
)

figura.show()

#Group --------------------------
#group_Kmeans_results = evaluate_kfold_kmeans(front_labels, front_data, n_splits=2)

"""for fold_result in group_Kmeans_results:
    fig = generar_pca_3d_desde_fold(fold_result)
    fig.show()
"""
