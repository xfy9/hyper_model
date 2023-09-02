import pandas as pd
import numpy as np
import jieba
# files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
med_file = 'F://gbert_data//IN_SUMMARY_DRUG_DETAIL1.csv'
diag_file = 'F://gbert_data//IN_SUMMARY_DISCHARE_DIAG1.csv'
procedure_file = 'F://gbert_data//IN_MEDICAL_RECORD_OPERATION1.csv'
sym_file="F://gbert_data//ADMISSION_INFORMATION_RECORD7.csv"

def change_hadm(x):
    if '_' in x:
        return x.split('_',1)[0]
    if '-' in x:
        return x.split('-',1)[0]
    else:
        return x
def change_subject(x):
    if '_' in x:
        return x.split('_',1)[1]
    if '-' in x:
        return x.split('-',1)[1]
    else:
        return x
def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    #     pro_pd = pro_pd[pro_pd['SEQ_NUM']<5]
    #     def icd9_tree(x):
    #         if x[0]=='E':
    #             return x[:4]
    #         return x[:3]
    #     pro_pd['ICD9_CODE'] = pro_pd['ICD9_CODE'].map(icd9_tree)
    pro_pd['SUBJECT_ID'] = pro_pd['SUBJECT_ID'].apply(change_subject)
    # pro_pd['HADM_ID'] = pro_pd['HADM_ID'].apply(change_hadm)

    pro_pd['SUBJECT_ID'] = pro_pd['SUBJECT_ID'].astype(str)
    pro_pd['HADM_ID'] = pro_pd['HADM_ID'].astype(str)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    # pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd

def process_med():
    print('process_med')
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # filter
    med_pd['SUBJECT_ID'] = med_pd['SUBJECT_ID'].apply(change_subject)
    # med_pd['HADM_ID'] = med_pd['HADM_ID'].apply(change_hadm)
    med_pd.drop(index=med_pd[med_pd['HADM_ID'] == ' '].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)


    # def func(x):
    #     try:
    #         return pd.to_datetime(
    #     med_pd['STARTDATE'], format='%d/%m/%Y %H:%M:%S')
    #     except:
    #         return np.NaN
    # print("start")
    # med_pd['STARTDATE'].apply(lambda x: func(x))
    # print("end")
    # med_pd['STARTDATE'] = pd.to_datetime(
    #     med_pd['STARTDATE'], format='%d/%m/%Y %H:%M:%S')
    med_pd.dropna(inplace=True)
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID',
                           'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    print(med_pd)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID', 'HADM_ID', 'STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new

    med_pd = filter_first24hour_med(med_pd)
    #     med_pd = med_pd.drop(columns=['STARTDATE'])

    # med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # visit > 2
    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    return med_pd.reset_index(drop=True)
# 创建停用词列表
def stopwordslist():
    # stopwords = [line.strip() for line in open('F:\\gbert_data\\stop_word.txt',encoding='UTF-8').readlines()]
    stopwords = ["，","：","。","、"]
    return stopwords
def cut_str(x):
    y=jieba.cut(x)
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in y:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ","
    return outstr


def process_sym():
    print('process_sym')
    sym_pd = pd.read_csv(sym_file).astype(str)
    # sym_pd['HADM_ID'] = sym_pd['HADM_ID'].apply(change_hadm)
    sym_pd.drop(index=sym_pd[sym_pd['CHIEF_COMPLAINTS'] == ' '].index, axis=0, inplace=True)
    # sym_pd['CHIEF_COMPLAINTS']=jieba.cut(sym_pd['CHIEF_COMPLAINTS'])
    sym_pd.dropna(subset=['CHIEF_COMPLAINTS'])
    sym_pd['CHIEF_COMPLAINTS'] = sym_pd['CHIEF_COMPLAINTS'].apply(cut_str)
    return sym_pd.reset_index(drop=True)

def process_diag():
    print('process_diad')
    diag_pd = pd.read_csv(diag_file)
    # filter

    diag_pd['SUBJECT_ID'] = diag_pd['SUBJECT_ID'].apply(change_subject)
    # diag_pd['HADM_ID'] = diag_pd['HADM_ID'].apply(change_hadm)
    diag_pd.drop(index=diag_pd[diag_pd['HADM_ID'] == ' '].index, axis=0, inplace=True)
    diag_pd.drop(index=diag_pd[diag_pd['ICD9_CODE'] == '--'].index, axis=0, inplace=True)
    diag_pd.dropna(inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    # print(diag_pd)
    return diag_pd.reset_index(drop=True)

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]

    return pro_pd.reset_index(drop=True)


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}).sort_values(
        by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'],
                                                                                                         ascending=False).reset_index(
        drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]

    return med_pd.reset_index(drop=True)

# def change(x):
#     if x.isnumeric():
#         return x
def process_all():
    sym_pd = process_sym()
    # get med and diag (visit>=2)
    med_pd = process_med()
    # med_pd = ndc2atc4(med_pd)
    #     med_pd = filter_300_most_med(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()
    #     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    sym_pd_key=sym_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    print(med_pd_key.dtypes)
    print(diag_pd_key.dtypes)
    print(pro_pd_key.dtypes)


    diag_pd_key.dropna(inplace=True)
    med_pd_key.dropna(inplace=True)
    # pro_pd_key['HADM_ID'] = pro_pd_key['HADM_ID'].apply(change)
    pro_pd_key.dropna(inplace=True)
    sym_pd_key.dropna(inplace=True)
    # diag_pd_key['HADM_ID'] = diag_pd_key['HADM_ID'].astype(dtype=str)
    # diag_pd_key['HADM_ID'] = diag_pd_key['HADM_ID'].astype(dtype=str)
    # med_pd['SUBJECT_ID'] = med_pd['SUBJECT_ID'].apply(int)
    print(diag_pd_key.dtypes)
    diag_pd_key = diag_pd_key.astype(dtype=str)
    med_pd_key = med_pd_key.astype(dtype=str)
    sym_pd_key = sym_pd_key.astype(dtype=str)
    print(diag_pd_key.dtypes)
    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(sym_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    print(med_pd.dtypes)
    med_pd['HADM_ID'] = med_pd['HADM_ID'].astype(dtype=str)
    sym_pd['HADM_ID'] = sym_pd['HADM_ID'].astype(dtype=str)
    sym_pd = sym_pd.astype(dtype=str)
    print(med_pd.dtypes)
    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    sym_pd = sym_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(
        columns={'ICD9_CODE': 'PRO_CODE'})
    sym_pd=sym_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['CHIEF_COMPLAINTS'].unique().reset_index()
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    sym_pd['CHIEF_COMPLAINTS']=sym_pd['CHIEF_COMPLAINTS'].map(lambda x: list(x))

    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(sym_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data

def statistics():
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag = 0
    avg_med = 0
    avg_pro = 0
    max_diag = 0
    max_med = 0
    max_pro = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        z = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x = set(x)
        y = set(y)
        z = set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)


data = process_all()
statistics()
data.to_pickle('data_final.pkl')
data.head()