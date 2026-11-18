import numpy as np 

#internal lib
from Basic_Functions.reading_functions_fromDB import *
from Basic_Functions.Baseline_correction import *
from Basic_Functions.PCA import *  

#-------------------------------------- Data --------------------------------------------
references = r"C:\licha\MachineL_for_Greenstones\Database\JADE_mapping_reflectance.json"

#-------------------------------------- cleaning and separating data --------------------
complete_DB = load_DB_from_JSON(references)
removing_motagua = removing_specific_datafrom_label(complete_DB, "Motagua") #Because it doesn't have the same polished surface

front_data = removing_specific_datafrom_label(removing_motagua, "reverso") #len: 192
back_data = removing_specific_datafrom_label( removing_motagua, "frente")#len: 191

#--------------------------------retriving labels ---------------------------------------

front_labels, front_data = retriving_labels_and_data_from_DB(front_data)
back_labels, back_data  = retriving_labels_and_data_from_DB(back_data)

print("front labels: ", front_labels[:10])
print("back labels:", back_labels[:10])

#--------------------------------making baseline correction------------------------------

front_sliced_data = slicing_all_BD_wavenumber(front_data, 4000, 400, 1200, 400)
back_sliced_data = slicing_all_BD_wavenumber(back_data, 4000, 400, 1200, 400)


