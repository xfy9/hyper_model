# -*- coding: UTF8 -*-

# cPickle是python2系列用的，3系列已经不用了，直接用pickle就好了
import pickle
def avg(values,len):
    sum_values=0
    for i in values:
        sum_values=sum_values+float(i)
    return sum_values/len



def statistics(data):
    print('#patients ', data['PERSON_INFO_ID'].unique().shape)
    print('#clinical events',len(data))
    # print(len(data))

    AGE=data['AGE'].values
    BODY_TEMPERATURE=data['BODY_TEMPERATURE'].values
    PULSE_RATE=data['PULSE_RATE'].values
    BREATHING_RATE=data['BREATHING_RATE'].values
    LSBP=data['LSBP'].values
    LDBP=data['LDBP'].values
    RSBP=data['RSBP'].values
    RDBP=data['RDBP'].values
    HEIGHT=data['HEIGHT'].values
    WEIGHT=data['WEIGHT'].values
    WAIST=data['WAIST'].values
    BMI=data['BMI'].values
    EXERCISE_FREQ_CODE=data['EXERCISE_FREQ_CODE'].values
    SMOKING_STATUS_CODE=data['SMOKING_STATUS_CODE'].values
    DRINKING_FREQ_CODE=data['DRINKING_FREQ_CODE'].values
    HEART_RATE=data['HEART_RATE'].values
    TCHO=data['TCHO'].values
    TG=data['TG'].values
    LDLC=data['LDLC'].values
    HDLC=data['HDLC'].values
    IS_HYPER_3YEARS=data['IS_HYPER_3YEARS'].values
    # diag = data['ICD9_CODE'].values
    # med = data['ATC4'].values

    # unique_diag = set([i for i in diag])
    # unique_med = set([j for i in med for j in list(i)])

    unique_AGE = set([i for i in AGE])
    unique_BODY_TEMPERATURE = set([i for i in BODY_TEMPERATURE])
    unique_PULSE_RATE = set([i for i in PULSE_RATE])
    unique_BREATHING_RATE = set([i for i in BREATHING_RATE])
    unique_LSBP = set([i for i in LSBP])
    unique_LDBP = set([i for i in LDBP])
    unique_RSBP = set([i for i in RSBP])
    unique_RDBP = set([i for i in RDBP])
    unique_HEIGHT = set([i for i in HEIGHT])
    unique_WEIGHT = set([i for i in WEIGHT])
    unique_WAIST = set([i for i in WAIST])
    unique_BMI = set([i for i in BMI])
    unique_EXERCISE_FREQ_CODE = set([i for i in EXERCISE_FREQ_CODE])
    unique_SMOKING_STATUS_CODE = set([i for i in SMOKING_STATUS_CODE])
    unique_DRINKING_FREQ_CODE = set([i for i in DRINKING_FREQ_CODE])
    unique_HEART_RATE = set([i for i in HEART_RATE])
    unique_TCHO = set([i for i in TCHO])
    unique_TG = set([i for i in TG])
    unique_LDLC = set([i for i in LDLC])
    unique_HDLC = set([i for i in HDLC])
    unique_IS_HYPER_3YEARS = set([i for i in IS_HYPER_3YEARS])

    # print('#AGE ', len(unique_AGE))
    # print('#BODY_TEMPERATURE ', len(unique_BODY_TEMPERATURE))
    # print('#unique_PULSE_RATE ', len(unique_PULSE_RATE))
    # print('#unique_BREATHING_RATE ', len(unique_BREATHING_RATE))
    # print('#unique_LSBP ', len(unique_LSBP))
    # print('#unique_LDBP ', len(unique_LDBP))
    # print('#unique_RSBP ', len(unique_RSBP))
    # print('#unique_RDBP ', len(unique_RDBP))
    # print('#unique_HEIGHT ', len(unique_HEIGHT))
    # print('#unique_WEIGHT ', len(unique_WEIGHT))
    # print('#unique_WAIST ', len(unique_WAIST))
    # print('#unique_BMI ', len(unique_BMI))
    # print('#unique_EXERCISE_FREQ_CODE ', len(unique_EXERCISE_FREQ_CODE))
    # print('#unique_SMOKING_STATUS_CODE ', len(unique_SMOKING_STATUS_CODE))
    # print('#unique_DRINKING_FREQ_CODE ', len(unique_DRINKING_FREQ_CODE))
    # print('#unique_HEART_RATE ', len(unique_HEART_RATE))
    # print('#unique_TCHO ', len(unique_TCHO))
    # print('#unique_TG ', len(unique_TG))
    # print('#unique_LDLC ', len(unique_LDLC))
    # print('#unique_HDLC ', len(unique_HDLC))
    # print('#unique_IS_HYPER_3YEARS ', len(unique_IS_HYPER_3YEARS))

    # unique_AGE = set([i for i in AGE])
    avg_AGE=avg(AGE,len(data))
    avg_BODY_TEMPERATURE = avg(BODY_TEMPERATURE,len(data))
    avg_PULSE_RATE = avg(PULSE_RATE,len(data))
    avg_BREATHING_RATE = avg(BREATHING_RATE,len(data))
    avg_LSBP = avg(LSBP,len(data))
    avg_LDBP = avg(LDBP,len(data))
    avg_RSBP = avg(RSBP,len(data))
    avg_RDBP = avg(RDBP,len(data))
    avg_HEIGHT = avg(HEIGHT,len(data))
    avg_WEIGHT = avg(WEIGHT,len(data))
    avg_WAIST = avg(WAIST,len(data))
    avg_BMI = avg(BMI,len(data))
    avg_EXERCISE_FREQ_CODE = avg(EXERCISE_FREQ_CODE,len(data))
    avg_SMOKING_STATUS_CODE = avg(SMOKING_STATUS_CODE,len(data))
    avg_DRINKING_FREQ_CODE = avg(DRINKING_FREQ_CODE,len(data))
    avg_HEART_RATE = avg(HEART_RATE,len(data))
    avg_TCHO = avg(TCHO,len(data))
    avg_TG = avg(TG,len(data))
    avg_LDLC = avg(LDLC,len(data))
    avg_HDLC = avg(HDLC,len(data))
    avg_IS_HYPER_3YEARS = avg(IS_HYPER_3YEARS,len(data))

    print("#avg_AGE",avg_AGE)
    print("#avg_BODY_TEMPERATURE",avg_BODY_TEMPERATURE)
    print("#avg_PULSE_RATE",avg_PULSE_RATE)
    print("#avg_BREATHING_RATE",avg_BREATHING_RATE)
    print("#avg_LSBP",avg_LSBP)
    print("#avg_LDBP",avg_LDBP)
    print("#avg_RSBP",avg_RSBP)
    print("#avg_RDBP",avg_RDBP)
    print("#avg_HEIGHT",avg_HEIGHT)
    print("#avg_WEIGHT",avg_WEIGHT)
    print("#avg_WAIST",avg_WAIST)
    print("#avg_BMI",avg_BMI)
    print("#avg_EXERCISE_FREQ_CODE",avg_EXERCISE_FREQ_CODE)
    print("#avg_SMOKING_STATUS_CODE",avg_SMOKING_STATUS_CODE)
    print("#avg_DRINKING_FREQ_CODE",avg_DRINKING_FREQ_CODE)
    print("#avg_HEART_RATE",avg_HEART_RATE)
    print("#avg_TCHO",avg_TCHO)
    print("#avg_TG",avg_TG)
    print("#avg_LDLC",avg_LDLC)
    print("#avg_HDLC",avg_HDLC)
    print("#avg_IS_HYPER_3YEARS",avg_IS_HYPER_3YEARS)





    # avg_diag = 0
    # avg_med = 0
    # max_diag = 0
    # max_med = 0
    # cnt = 0
    # max_visit = 0
    # avg_visit = 0
    #
    # for subject_id in data['SUBJECT_ID'].unique():
    #     item_data = data[data['SUBJECT_ID'] == subject_id]
    #     x = []
    #     y = []
    #     visit_cnt = 0
    #     for index, row in item_data.iterrows():
    #         visit_cnt += 1
    #         cnt += 1
    #         x.extend(list(row['ICD9_CODE']))
    #         y.extend(list(row['ATC4']))
    #     x = set(x)
    #     y = set(y)
    #     avg_diag += len(x)
    #     avg_med += len(y)
    #     avg_visit += visit_cnt
    #     if len(x) > max_diag:
    #         max_diag = len(x)
    #     if len(y) > max_med:
    #         max_med = len(y)
    #     if visit_cnt > max_visit:
    #         max_visit = visit_cnt
    #
    # print('#avg of diagnoses ', avg_diag / cnt)
    # print('#avg of medicines ', avg_med / cnt)
    # print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    # print('#max of diagnoses ', max_diag)
    # print('#max of medicines ', max_med)
    # print('#max of visit ', max_visit)

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
def load_ids(data,file_name):
    ids = []
    with open(file_name, 'r') as f:
        for line in f:
            ids.append(line.rstrip('\n'))
    return data[data['PERSON_INFO_ID'].isin(ids)].reset_index(drop=True)
# def load_ids_1(data, data_id_1,file_name):
#     ids = []
#     with open(file_name, 'r') as f:
#         for line in f:
#             ids.append(line.rstrip('\n'))
#     data_person=data[data['PERSON_INFO_ID'].isin(ids)]
#     return data_person[~data_person['PERSON_INFO_ID'].isin(data_id_1)].reset_index(drop=True)

f1 = open('./琼海市塔洋镇卫生院/hyper-multi-visit.pkl','rb')
data1 = pickle.load(f1)
f2 = open('./琼海市塔洋镇卫生院/hyper-single-visit.pkl','rb')
data2 = pickle.load(f2)
#print(data1)
#print(data2)
#print('#patients ', data1['PERSON_INFO_ID'].unique().shape)
###############zsw
print("muti-visit ")
data1_is_hyper_0  = data1[data1['IS_HYPER_3YEARS']==0]
data1_is_hyper_1  = data1[data1['IS_HYPER_3YEARS']==1]
print('#muti-visit patients set 0 ', data1_is_hyper_0['PERSON_INFO_ID'].unique().shape)
print('#muti-visit patients set 1 ', data1_is_hyper_1['PERSON_INFO_ID'].unique().shape)
print("muti-visit patients len ",len(data1_is_hyper_0))
print("muti-visit patients len ",len(data1_is_hyper_1))
#statistics(data1_is_hyper_1)
print("sigle-vist ")
data2_is_hyper_0  = data2[data2['IS_HYPER_3YEARS']==0]
data2_is_hyper_1  = data2[data2['IS_HYPER_3YEARS']==1]
print('#sigle-vist patients set 0 ', data2_is_hyper_0['PERSON_INFO_ID'].unique().shape)
print('#sigle-vist patients set 1 ', data2_is_hyper_1['PERSON_INFO_ID'].unique().shape)
print("sigle-vist patients len ",len(data2_is_hyper_0))
print("sigle-vist patients len ",len(data2_is_hyper_1))

#data_is_hyper_0 = data1_is_hyper_0.append(data2_is_hyper_0)
#data_is_hyper_1 = data1_is_hyper_1.append(data2_is_hyper_1)
#statistics(data_is_hyper_0)
#statistics(data_is_hyper_1)
###############zsw
# data_is_hyper_0=data[data['IS_HYPER_3YEARS']==0]
# data_is_hyper_1=data[data['IS_HYPER_3YEARS']==1]

# data_test=load_ids(data1,'hyper/hyper-test-id.txt')
# print(data_test)
# data_test_1_ids=data_test[data_test['IS_HYPER_3YEARS']==1]['PERSON_INFO_ID'].values
# data_test_1=data_test[data_test['PERSON_INFO_ID'].isin(data_test_1_ids)].reset_index(drop=True)
# len=len(data_test_1[data_test_1['IS_HYPER_3YEARS']==0])
#
# data_test_2=data_test[~data_test['PERSON_INFO_ID'].isin(data_test_1_ids)].reset_index(drop=True)
# data_test_2.sort_values(by="PERSON_INFO_ID" , inplace=True, ascending=True)
# data_test_2=data_test_2[0:318]
# data_test_1_2=data_test_1.append(data_test_2)


def ls2file(list_data, file_name):
    with open(file_name, 'w') as fout:
        for item in list_data:
            fout.write(str(item) + '\n')
# ls2file(data_test_1_2['PERSON_INFO_ID'].values, 'hyper/hyper-test-id2.txt')
# data=data1.append(data2)
# data_is_hyper_0=data[data['IS_HYPER_3YEARS']==0]
# data_is_hyper_1=data[data['IS_HYPER_3YEARS']==1]
# # for i in data['BREATHING_RATE']:
# #     if i=="74":
# #         print(i)
# # print(data_is_hyper_0)
# print("is_hyper=0\n")
# statistics(data_is_hyper_0)
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print("is_hyper=1\n")
# statistics(data_is_hyper_1)
# print("=====================")
# statistics(data1)
# statistics(data2)
