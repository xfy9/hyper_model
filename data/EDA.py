# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import dill
from random import shuffle
# import jieba
import random
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'data'))
    print(os.getcwd())
except:
    pass

# %%
import pandas as pd
from collections import defaultdict
import numpy as np

diag_file = 'F://gbert_data//IN_SUMMARY_DISCHARE_DIAG1.csv'
med_file='F://gbert_data//IN_SUMMARY_DRUG_DETAIL1.csv'
sym_file="F://gbert_data//ADMISSION_INFORMATION_RECORD7.csv"


def process_diag():
    print('process_diad')
    diag_pd = pd.read_csv(diag_file)
    # filter
    # diag_pd.drop(columns =['OUTPATIENT_DIAG_NAME'], axis=1, inplace=True)
    diag_pd.drop(index=diag_pd[diag_pd['HADM_ID'] == ''].index, axis=0, inplace=True)
    diag_pd.dropna(inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    # print(diag_pd)
    return diag_pd.reset_index(drop=True)
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
# 创建停用词列表
def stopwordslist():
    # stopwords = [line.strip() for line in open('F:\\gbert_data\\stop_word.txt',encoding='UTF-8').readlines()]
    stopwords = ["，","：","。","、"]
    return stopwords
# def cut_str(x):
#     jieba.load_userdict('F:\\gbert_data\\dict.txt')
#     y=jieba.cut(x)
#     # 创建一个停用词列表
#     stopwords = stopwordslist()
#     # 输出结果为outstr
#     outstr = ''
#     # 去停用词
#     for word in y:
#         if word not in stopwords:
#             if word != '\t':
#                 outstr += word
#                 outstr += ","
#     return outstr
def process_sym():
    print('process_sym')
    sym_pd = pd.read_csv(sym_file).astype(str)
    # sym_pd['HADM_ID'] = sym_pd['HADM_ID'].apply(change_hadm)
    # sym_pd.drop(index=sym_pd[sym_pd['HADM_ID'] == ' '].index, axis=0, inplace=True)
    # sym_pd['CHIEF_COMPLAINTS']=jieba.cut(sym_pd['CHIEF_COMPLAINTS'])
    sym_pd.drop(index=sym_pd[sym_pd['CHIEF_COMPLAINTS'] == ' '].index, axis=0, inplace=True)
    # sym_pd['CHIEF_COMPLAINTS']=jieba.cut(sym_pd['CHIEF_COMPLAINTS'])
    sym_pd.dropna(subset=['CHIEF_COMPLAINTS'])
    sym_pd['CHIEF_COMPLAINTS'] = sym_pd['CHIEF_COMPLAINTS'].apply(cut_str)
    return sym_pd.reset_index(drop=True)
def process_med():
    print('process_med')
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'})
    # filter
    # med_pd['SUBJECT_ID'] = med_pd['SUBJECT_ID'].apply(change_subject)
    med_pd.drop(index=med_pd[med_pd['HADM_ID'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    # med_pd['STARTDATE'] = pd.to_datetime(
    #     med_pd['STARTDATE'], format='%d/%m/%Y %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID',
                           'STARTDATE'], inplace=True)
    med_pd.drop(columns=['STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    print(med_pd)


    # def filter_first24hour_med(med_pd):
    #     med_pd_new = med_pd.drop(columns=['NDC'])
    #     med_pd_new = med_pd_new.groupby(
    #         by=['SUBJECT_ID', 'HADM_ID'])
    #     # print("first24hour")
    #     # print(med_pd_new.groupby(
    #     #     by=['SUBJECT_ID', 'HADM_ID']).head([1]))
    #     med_pd_new = pd.merge(med_pd_new, med_pd, on=[
    #                           'SUBJECT_ID', 'HADM_ID', 'STARTDATE'])
    #     med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
    #     return med_pd_new
    # med_pd = filter_first24hour_med(med_pd)  # or next line
#     med_pd = med_pd.drop(columns=['STARTDATE'])

    med_pd = med_pd.drop_duplicates()
    print(med_pd)
    return med_pd.reset_index(drop=True)
def filter_by_visit_range(data_pd, v_range=(1, 2)):
    a = data_pd[['SUBJECT_ID', 'HADM_ID']].groupby(
        by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
    a = a[(a['HADM_ID_Len'] >= v_range[0]) & (a['HADM_ID_Len'] < v_range[1])]
    data_pd_filter = a.reset_index(drop=True)
    data_pd = data_pd.merge(
        data_pd_filter[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')
    return data_pd.reset_index(drop=True)
def filter_diag(diag_pd, num=128):
    print('filter diag')
    # diag_pd.rename(columns={'NDC': 'ATC4'}, inplace=True)
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(
        diag_count.loc[:num, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)
def change(x):
    if x.isnumeric():
        return x

def process_all(visit_range=(1, 2)):
    sym_pd = process_sym()
    sym_pd = filter_by_visit_range(sym_pd, visit_range)
    # get med and diag (visit>=2)
    med_pd = process_med()
    # med_pd = ndc2atc4(med_pd)
#     med_pd = filter_300_most_med(med_pd)


    diag_pd = process_diag()
    diag_pd = filter_diag(diag_pd, num=1999)

#     side_pd = process_side()

#     pro_pd = process_procedure()
#     pro_pd = filter_1000_most_pro(pro_pd)

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    sym_pd_key = sym_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
#     pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
#     side_pd_key = side_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    print(med_pd_key.dtypes)
    print(diag_pd_key.dtypes)
    # diag_pd_key1 = diag_pd_key['HADM_ID'].apply(change)
    # diag_pd_key['HADM_ID'] = diag_pd_key['HADM_ID'].apply(change_subject)
    # diag_pd_key['HADM_ID'] = diag_pd_key['HADM_ID'].apply(change)
    diag_pd_key.dropna(inplace=True)
    diag_pd_key['HADM_ID'] = diag_pd_key['HADM_ID'].astype(dtype=str)
    # med_pd['SUBJECT_ID'] = med_pd['SUBJECT_ID'].apply(int)
    print(diag_pd_key.dtypes)
    diag_pd_key = diag_pd_key.astype(dtype=str)
    med_pd_key = med_pd_key.astype(dtype=str)
    sym_pd_key = sym_pd_key.astype(dtype=str)
    print(diag_pd_key.dtypes)
    combined_key = med_pd_key.merge(
        diag_pd_key, on=[ 'SUBJECT_ID','HADM_ID'], how='inner')
    combined_key = combined_key.merge(
        sym_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     combined_key = combined_key.merge(side_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(
        combined_key, on=[  'SUBJECT_ID','HADM_ID'], how='inner')
    med_pd = med_pd.astype(dtype=str)

    med_pd = med_pd.merge(
        combined_key, on=[ 'SUBJECT_ID','HADM_ID'], how='inner')
    sym_pd = sym_pd.merge(
        combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     side_pd = side_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'ICD9_CODE'].unique().reset_index()
    med_pd.rename(columns={'NDC': 'ATC4'}, inplace=True)
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'ATC4'].unique().reset_index()
    sym_pd = sym_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])[
        'CHIEF_COMPLAINTS'].unique().reset_index()
#     pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})
    diag_pd['ICD9_CODE'] = diag_pd['ICD9_CODE'].map(lambda x: list(x))
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: list(x))
    sym_pd['CHIEF_COMPLAINTS'] = sym_pd['CHIEF_COMPLAINTS'].map(lambda x: list(x))
#     pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=[ 'SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(sym_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data = data.merge(side_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
#     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
#     data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data
def filter_patient(data, dx_range=(2, np.inf), rx_range=(2, np.inf),sym_range=(2, np.inf)):
    print('filter_patient')
    # data['SUBJECT_ID']=data['HADM_ID']
    drop_subject_ls = []
    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]

        for index, row in item_data.iterrows():
            dx_len = len(list(row['ICD9_CODE']))
            rx_len = len(list(row['ATC4']))
            sym_len=len(list(row['CHIEF_COMPLAINTS']))
            if dx_len < dx_range[0] or dx_len > dx_range[1]:
                drop_subject_ls.append(subject_id)
                break
            if rx_len < rx_range[0] or rx_len > rx_range[1]:
                drop_subject_ls.append(subject_id)
                break
            if sym_len < sym_range[0] or sym_len > sym_range[1]:
                drop_subject_ls.append(subject_id)
                break
    # data.drop(index=data[data['SUBJECT_ID'].isin(
    #     drop_subject_ls)].index, axis=0, inplace=True)
    return data.reset_index(drop=True)
def load_gamenet_multi_visit_data_with_pro(file_name='data_final.pkl'):
    data = pd.read_pickle(file_name)
    data.rename(columns={'NDC': 'ATC4'}, inplace=True)
    data.drop(columns=['NDC_Len'], axis=1, inplace=True)

    # unique code save
    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values
    pro = data['PRO_CODE'].values
    sym=data['CHIEF_COMPLAINTS'].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    un_sym = set([j for i in sym for j in list(i)])
    unique_sym=set()
    for i in un_sym:
        for j in range(len(i.split(','))):
            unique_sym.add(i.split(',')[j])

    return data, unique_pro, unique_diag, unique_med, unique_sym
def run(visit_range=(1, 2)):
    data = process_all(visit_range)
    data = filter_patient(data)

    # unique code save
    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values
    sym=data['CHIEF_COMPLAINTS'].values
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    un_sym = set([j for i in sym for j in list(i)])
    unique_sym = set()
    for i in un_sym:
        for j in range(len(i.split(','))):
            unique_sym.add(i.split(',')[j])
    return data, unique_diag, unique_med,unique_sym
def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['ATC4'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))

    avg_diag = 0
    avg_med = 0
    max_diag = 0
    max_med = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['ATC4']))
        x = set(x)
        y = set(y)
        avg_diag += len(x)
        avg_med += len(y)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of visit ', max_visit)

def main():
    # print('-' * 20 + '\ndata-multi processing ')
    # data_multi_visit, pro, diag2, med2, sym2 = load_gamenet_multi_visit_data_with_pro()
    #
    # with open('sx-vocab-multi.txt', 'w',encoding='utf-8') as fout:
    #     for code in sym2:
    #         fout.write(code + '\n')
    print('-'*20 + '\ndata-single processing')
    data_single_visit, diag1, med1,sym1 = run(visit_range=(1, 2))
    print('-'*20 + '\ndata-multi processing ')
    data_multi_visit, pro, diag2, med2,sym2 = load_gamenet_multi_visit_data_with_pro()
#     med_diag_pair = gen_med_diag_pair(data)

    unique_diag = diag1 | diag2
    unique_med = med1 | med2
    unique_sym = sym1 | sym2
    with open('dx-vocab.txt', 'w') as fout:
        for code in unique_diag:
            fout.write(code + '\n')
    with open('rx-vocab.txt', 'w') as fout:
        for code in unique_med:
            fout.write(code + '\n')
    with open('sx-vocab.txt', 'w',encoding='utf-8') as fout:
        for code in unique_sym:
            fout.write(code + '\n')

    with open('rx-vocab-multi.txt', 'w') as fout:
        for code in med2:
            fout.write(code + '\n')
    with open('dx-vocab-multi.txt', 'w') as fout:
        for code in diag2:
            fout.write(code + '\n')
    with open('sx-vocab-multi.txt', 'w',encoding='utf-8') as fout:
        for code in sym2:
            fout.write(code + '\n')
    with open('px-vocab-multi.txt', 'w') as fout:
        for code in pro:
            fout.write(str(code) + '\n')

    # save data
    data_single_visit.to_pickle('data-single-visit.pkl')
    data_multi_visit.to_pickle('data-multi-visit.pkl')

#     med_diag_pair.to_pickle('med_diag.pkl')
#     print('med2diag len:', len(med_diag_pair))

    print('-'*20 + '\ndata-single stat')
    statistics(data_single_visit)
    print('-'*20 + '\ndata_multi stat')
    statistics(data_multi_visit)

    split_dataset()
    return data_single_visit, data_multi_visit

random.seed(1203)
def split_dataset(data_path='data-multi-visit.pkl'):
    data = pd.read_pickle(data_path)
    sample_id = data['SUBJECT_ID'].unique()

    random_number = [i for i in range(len(sample_id))]
#     shuffle(random_number)

    train_id = sample_id[random_number[:int(len(sample_id)*2/3)]]
    eval_id = sample_id[random_number[int(
        len(sample_id)*2/3): int(len(sample_id)*5/6)]]
    test_id = sample_id[random_number[int(len(sample_id)*5/6):]]

    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')

    ls2file(train_id, 'train-id.txt')
    ls2file(eval_id, 'eval-id.txt')
    ls2file(test_id, 'test-id.txt')

    print('train size: %d, eval size: %d, test size: %d' %
          (len(train_id), len(eval_id), len(test_id)))
if __name__ == '__main__':
    split_dataset()
