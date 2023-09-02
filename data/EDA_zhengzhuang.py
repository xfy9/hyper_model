import pandas as pd

# files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
sym_file = 'F:\\gbert_data\\ADMISSION_INFORMATION_RECORD.csv'
output_sym_file="F:\\gbert_data\\ADMISSION_INFORMATION_RECORD1.csv"

sym_pd = pd.read_csv(sym_file)
sym_pd = sym_pd.drop(columns=['PATIENT_ID'])