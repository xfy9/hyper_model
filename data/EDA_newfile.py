import pandas as pd

# files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
med_file = 'F:\\gbert_data\\IN_SUMMARY_DRUG_DETAIL1.csv'
output_med_file="IN_SUMMARY_DRUG_DETAIL1.csv"

diag_file = 'IN_SUMMARY_DISCHARE_DIAG.csv'
output_diag_file="IN_SUMMARY_DISCHARE_DIAG1.csv"
procedure_file = 'IN_MEDICAL_RECORD_OPERATION.csv'
output_procedure_file="F:\\gbert_data\\IN_SUMMARY_DRUG_DETAIL2.csv"
output_procedure_file1="F:\\gbert_data\\IN_MEDICAL_RECORD_OPERATION.csv"

sym_file = 'F:\\gbert_data\\ADMISSION_INFORMATION_RECORD.csv'
output_sym_file="F:\\gbert_data\\ADMISSION_INFORMATION_RECORD7.csv"
def readwrite():
    date_1=pd.read_csv(diag_file,header=0,sep=',')
    date_1[['PATIENT_ID', 'INPATIENT_FORM_NO','OUTPATIENT_DIAG_CODE']].to_csv(output_diag_file, sep=',', header=True,index=False)
def readwrite1():
    date_1=pd.read_csv(med_file,header=0,sep=',',memory_map=True)
    date_1[['PATIENT_ID', 'INPATIENT_FORM_NO','PRESCRIPTION_DATE','DRUG_STANDARD_CODE']].to_csv(output_med_file, sep=',', header=True,index=False)
def readwrite2():
    date_1=pd.read_csv(procedure_file,header=0,sep=',')
    date_1[['PATIENT_ID', 'INPATIENT_FORM_NO','OPERATION_CODE']].to_csv(output_procedure_file1, sep=',', header=True,index=False)


def readwrite3():
    date_1=pd.read_csv(med_file,header=0,sep=',',chunksize=10000)
    for data in date_1:
        data.rename(columns={'PATIENT_ID': 'SUBJECT_ID'}, inplace=True)
        data.rename(columns={'INPATIENT_FORM_NO': 'HADM_ID'}, inplace=True)
        data.rename(columns={'PRESCRIPTION_DATE': 'STARTDATE'}, inplace=True)
        data.rename(columns={'DRUG_STANDARD_CODE': 'NDC'}, inplace=True)
        data.to_csv(output_procedure_file, sep=',',mode="a",header=True, index=False)

def readwrite4():
    date_1=pd.read_csv(sym_file,header=0,sep=',',memory_map=True)
    # date_1=date_1.dropna(subset=['CHIEF_COMPLAINTS', 'CURRENT_DISEASE','DISEASE_HISTORY','SPEC_SITUATION','ASSIST_EXAM_RESULT'])
    date_1[['PATIENT_ID', 'INPATIENT_FORM_NO','CHIEF_COMPLAINTS']].to_csv(output_sym_file, sep=',', header=True,index=False)

def readwrite5():
    date_1=pd.read_csv("F:\\gbert_data\\ADMISSION_INFORMATION_RECORD3.csv",header=0,sep=',',memory_map=True)
    date_1[['PATIENT_ID', 'INPATIENT_FORM_NO','CHIEF_COMPLAINTS']].to_csv("F:\\gbert_data\\ADMISSION_INFORMATION_RECORD4.csv", sep=',', header=True,index=False)

def main():
    readwrite4()

if __name__ == '__main__':
    main()