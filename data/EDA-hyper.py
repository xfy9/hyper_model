# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import sys

import dill
from random import shuffle
# import jieba
import random
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'data/hyper'))
    print(os.getcwd())
except:
    pass

# %%
import pandas as pd
from collections import defaultdict
import numpy as np

# multi_last_file="E:\wmz\hyper-bert-multi-task-new(-ishyper)\data\HYPER_FULL_EXTRACT_NEW_MULTI.csv"
#multi_last_file="./HYPER_FULL_EXTRACT_NEW_MULTI.csv"
# multi_last_file="./琼海市塔洋镇卫生院/均衡数据/多次均衡数据.csv"
multi_last_file= "gonggong/multi.csv"

# multi_positive_file="E:\wmz\hyper-bert-no-multi-task\data\HYPER_POSITIVE_ORDER_EXTRACT.csv"
# multi_negative_file="E:\wmz\hyper-bert-no-multi-task\data\HYPER_NEGATIVE_ORDER_EXTRACT.csv"
#single_file="E:\wmz\hyper-bert-multi-task-new(-ishyper)\data\HYPER_FULL_EXTRACT_NEW_SINGLE.csv"
#single_file="./HYPER_FULL_EXTRACT_NEW_SINGLE.csv"
# single_file="./琼海市塔洋镇卫生院/均衡数据/琼海市塔洋镇卫生院_单次.csv"
single_file= "gonggong/single.csv"

path = "gonggong/"


# 获取每个模型
def get_person_info_count():
    count_1 = 0
    hyper_pd = pd.read_csv('./HYPER_FULL_EXTRACT_NEW_MULTI.csv', header=None)
    hyper_pd.rename(columns={1: 'PERSON_INFO_ID'}, inplace=True)
    person_info_value = hyper_pd['PERSON_INFO_ID'].values
    print(person_info_value[0])
    count_dic = {}
    for person in person_info_value:
        if person in count_dic.keys():
            count_dic[person]+=1
        else:
            count_dic[person]=1
    for p_key in count_dic.keys():
        if count_dic[p_key]>15:
            print(p_key,'   ',count_dic[p_key])
            #count_1+=1
    # print("11111:  ",count_1)
    #34508
    print('count_key',len(count_dic.keys()))



def process_file(file):
    # hyper_pd = pd.read_csv(multi_positive_file)
    hyper_pd = pd.read_csv(file, header=None)
    hyper_pd.rename(columns={0: 'HEALTH_EXAM_ID'}, inplace=True)
    hyper_pd.rename(columns={1: 'PERSON_INFO_ID'}, inplace=True)
    hyper_pd.rename(columns={2: 'AGE'}, inplace=True)
    hyper_pd.rename(columns={3: 'BODY_TEMPERATURE'}, inplace=True)
    hyper_pd.rename(columns={4: 'PULSE_RATE'}, inplace=True)
    hyper_pd.rename(columns={5: 'BREATHING_RATE'}, inplace=True)
    hyper_pd.rename(columns={6: 'LSBP'}, inplace=True)
    hyper_pd.rename(columns={7: 'LDBP'}, inplace=True)
    hyper_pd.rename(columns={8: 'RSBP'}, inplace=True)
    hyper_pd.rename(columns={9: 'RDBP'}, inplace=True)
    hyper_pd.rename(columns={10: 'HEIGHT'}, inplace=True)
    hyper_pd.rename(columns={11: 'WEIGHT'}, inplace=True)
    hyper_pd.rename(columns={12: 'WAIST'}, inplace=True)
    hyper_pd.rename(columns={13: 'BMI'}, inplace=True)
    hyper_pd.rename(columns={14: 'EXERCISE_FREQ_CODE'}, inplace=True)
    hyper_pd.rename(columns={15: 'SMOKING_STATUS_CODE'}, inplace=True)
    hyper_pd.rename(columns={16: 'DRINKING_FREQ_CODE'}, inplace=True)
    hyper_pd.rename(columns={17: 'HEART_RATE'}, inplace=True)
    hyper_pd.rename(columns={18: 'TCHO'}, inplace=True)
    hyper_pd.rename(columns={19: 'TG'}, inplace=True)
    hyper_pd.rename(columns={20: 'LDLC'}, inplace=True)
    hyper_pd.rename(columns={21: 'HDLC'}, inplace=True)
    hyper_pd.rename(columns={22: 'IS_HYPER_3YEARS'}, inplace=True)

    # file_pd.drop(index=file_pd[file_pd['HADM_ID'] == '0'].index, axis=0, inplace=True)

    # hyper_pd.fillna(method='pad', inplace=True)

    #公共数据
    # hyper_pd = hyper_pd.astype(float)

    hyper_pd.fillna('0.0', inplace=True)
    hyper_pd.dropna(inplace=True)
    hyper_pd.drop_duplicates(inplace=True)
    # unique code save
    age_value=hyper_pd['AGE'].values
    body_tem_value = hyper_pd['BODY_TEMPERATURE'].values
    pulse_rate_value = hyper_pd['PULSE_RATE'].values
    breathing_rate_value = hyper_pd['BREATHING_RATE'].values
    lsbp_value = hyper_pd['LSBP'].values
    ldbp_value = hyper_pd['LDBP'].values
    rsbp_value = hyper_pd['RSBP'].values
    rdbp_value = hyper_pd['RDBP'].values
    height_value = hyper_pd['HEIGHT'].values
    weight_value = hyper_pd['WEIGHT'].values
    waist_value = hyper_pd['WAIST'].values
    bmi_value = hyper_pd['BMI'].values
    exercise_freq_value = hyper_pd['EXERCISE_FREQ_CODE'].values
    smoking_status_value = hyper_pd['SMOKING_STATUS_CODE'].values
    drinking_freq_value = hyper_pd['DRINKING_FREQ_CODE'].values
    heart_rate_value = hyper_pd['HEART_RATE'].values

    tcho_value = hyper_pd['TCHO'].values
    tg_value = hyper_pd['TG'].values
    ldlc_value = hyper_pd['LDLC'].values
    hdlc_value = hyper_pd['HDLC'].values

    is_hyper_value = hyper_pd['IS_HYPER_3YEARS'].values

    unique_age = set([i for i in age_value])
    unique_body_tem = set([i for i in body_tem_value])
    unique_pulse_rate = set([i for i in pulse_rate_value])
    unique_breathing_rate = set([i for i in breathing_rate_value])
    unique_lsbp = set([i for i in lsbp_value])
    unique_ldbp = set([i for i in ldbp_value])
    unique_rsbp = set([i for i in rsbp_value])
    unique_rdbp = set([i for i in rdbp_value])

    unique_height = set([i for i in height_value])
    unique_weight = set([i for i in weight_value])
    unique_waist = set([i for i in waist_value])
    unique_bmi = set([i for i in bmi_value])
    unique_exercise_freq = set([i for i in exercise_freq_value])
    unique_smoking_status = set([i for i in smoking_status_value])
    unique_drinking_freq = set([i for i in drinking_freq_value])
    unique_heart_rate = set([i for i in heart_rate_value])

    unique_tcho = set([i for i in tcho_value])
    unique_tg = set([i for i in tg_value])
    unique_ldlc = set([i for i in ldlc_value])
    unique_hdlc = set([i for i in hdlc_value])
    unique_is_hyper = set([i for i in is_hyper_value])
    return hyper_pd,unique_age,unique_body_tem,unique_pulse_rate,unique_breathing_rate,unique_lsbp,unique_ldbp,\
           unique_rsbp,unique_rdbp,unique_height,unique_weight,unique_waist,unique_bmi,unique_exercise_freq,unique_smoking_status,\
           unique_drinking_freq,unique_heart_rate,unique_tcho,unique_tg,unique_ldlc,unique_hdlc,unique_is_hyper

    print(hyper_pd)
def write_txt(txt_file,txt_data):
    with open(txt_file, 'w') as fout:
        for code in txt_data:
            fout.write(str(code) + '\n')

random.seed(1203)

# 正态分布划分正负中数据集
def split_dataset_zhengtai(path):
    hyper_pd = pd.read_csv(path, header=None)
    hyper_pd.rename(columns={0: 'HEALTH_EXAM_ID'}, inplace=True)
    hyper_pd.rename(columns={1: 'PERSON_INFO_ID'}, inplace=True)
    hyper_pd.rename(columns={2: 'AGE'}, inplace=True)
    hyper_pd.rename(columns={3: 'BODY_TEMPERATURE'}, inplace=True)
    hyper_pd.rename(columns={4: 'PULSE_RATE'}, inplace=True)
    hyper_pd.rename(columns={5: 'BREATHING_RATE'}, inplace=True)
    hyper_pd.rename(columns={6: 'LSBP'}, inplace=True)
    hyper_pd.rename(columns={7: 'LDBP'}, inplace=True)
    hyper_pd.rename(columns={8: 'RSBP'}, inplace=True)
    hyper_pd.rename(columns={9: 'RDBP'}, inplace=True)
    hyper_pd.rename(columns={10: 'HEIGHT'}, inplace=True)
    hyper_pd.rename(columns={11: 'WEIGHT'}, inplace=True)
    hyper_pd.rename(columns={12: 'WAIST'}, inplace=True)
    hyper_pd.rename(columns={13: 'BMI'}, inplace=True)
    hyper_pd.rename(columns={14: 'EXERCISE_FREQ_CODE'}, inplace=True)
    hyper_pd.rename(columns={15: 'SMOKING_STATUS_CODE'}, inplace=True)
    hyper_pd.rename(columns={16: 'DRINKING_FREQ_CODE'}, inplace=True)
    hyper_pd.rename(columns={17: 'HEART_RATE'}, inplace=True)
    hyper_pd.rename(columns={18: 'TCHO'}, inplace=True)
    hyper_pd.rename(columns={19: 'TG'}, inplace=True)
    hyper_pd.rename(columns={20: 'LDLC'}, inplace=True)
    hyper_pd.rename(columns={21: 'HDLC'}, inplace=True)
    hyper_pd.rename(columns={22: 'IS_HYPER_3YEARS'}, inplace=True)

    # file_pd.drop(index=file_pd[file_pd['HADM_ID'] == '0'].index, axis=0, inplace=True)
    hyper_pd.fillna(method='pad', inplace=True)
    hyper_pd.dropna(inplace=True)
    hyper_pd.drop_duplicates(inplace=True)

    sample_id = hyper_pd['PERSON_INFO_ID'].unique()
    random_number = [i for i in range(len(sample_id))]
    shuffle(random_number)

    # 总的训练+验证 训练+验证 80% 测试 20%
    train_eval_id = sample_id[random_number[:int(len(sample_id) * 1 / 5)]]
    # 总的测试
    test_id = sample_id[random_number[int(len(sample_id) * 1 / 5):]]

    random_train_eval_number = [i for i in range(len(train_eval_id))]
    shuffle(random_train_eval_number)
    # 用于一般训练的数据 30%
    normal_train_id = train_eval_id[random_train_eval_number[:int(len(train_eval_id) * 3 / 10)]]
    random_normal_train_number = [i for i in range(len(normal_train_id))]
    shuffle(random_normal_train_number)
    normal_train_train_id = normal_train_id[random_normal_train_number[:int(len(normal_train_id) * 4 / 5)]]
    normal_train_eval_id = normal_train_id[random_normal_train_number[int(len(normal_train_id) * 4 / 5):]]
    print('normal train size: %d' % len(normal_train_train_id))
    print('normal eval size: %d' % len(normal_train_eval_id))

    # 用于增量学习的数据 70%
    inc_train_eval_id = train_eval_id[random_train_eval_number[int(len(train_eval_id) * 3 / 10):]]

    print('test size: %d' % len(test_id))

    random_inc_train_eval_number =  [i for i in range(len(inc_train_eval_id))]
    shuffle(random_inc_train_eval_number)

    # 增量学习 分成7组训练+验证
    all_inc_train_eval_list = []
    for i in range(0,7):
        simple_train_eval_id = inc_train_eval_id[random_inc_train_eval_number[int(len(inc_train_eval_id) * i / 7):int(len(inc_train_eval_id) * (i+1) / 7)]]
        all_inc_train_eval_list.append(simple_train_eval_id)


    # 每单组再分为5份
    count = 0
    simple_train_id_list = []
    simple_eval_id_list = []
    for simple_train_eval_id in all_inc_train_eval_list:
        count+=1
        random_simple_train_eval_number = [i for i in range(len(simple_train_eval_id))]
        shuffle(random_simple_train_eval_number)
        simple_train_id = simple_train_eval_id[random_simple_train_eval_number[:int(len(simple_train_eval_id) * 4 / 5)]]
        simple_eval_id = simple_train_eval_id[random_simple_train_eval_number[int(len(simple_train_eval_id)*4/5):]]
        simple_train_id_list.append(simple_train_id)
        simple_eval_id_list.append(simple_eval_id)
        print('train size: %d, eval size: %d' %
              (len(simple_train_id), len(simple_eval_id)))

    return test_id,normal_train_train_id,normal_train_eval_id,simple_train_id_list,simple_eval_id_list


def all_split_dataset_zhengtai():

    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')
    test_id_pos, normal_train_train_id_pos, normal_train_eval_id_pos, simple_train_id_list_pos, simple_eval_id_list_pos = split_dataset_zhengtai("dongfan_data/HYPER_FULL_ORG_东方市港区中心卫生院_MULTI_pos.csv")
    test_id_nor, normal_train_train_id_nor, normal_train_eval_id_nor, simple_train_id_list_nor, simple_eval_id_list_nor = split_dataset_zhengtai("dongfan_data/HYPER_FULL_ORG_东方市港区中心卫生院_MULTI_nor.csv")
    test_id_neg, normal_train_train_id_neg, normal_train_eval_id_neg, simple_train_id_list_neg, simple_eval_id_list_neg = split_dataset_zhengtai("dongfan_data/HYPER_FULL_ORG_东方市港区中心卫生院_MULTI_neg.csv")
    #print("type ",test_id_neg)
    test_id = np.append(np.append(test_id_pos,test_id_nor),test_id_neg)
    #print("type ", test_id)
    normal_train_train_id = np.append(np.append(normal_train_train_id_pos,normal_train_train_id_nor),normal_train_train_id_neg)
    normal_train_eval_id = np.append(np.append(normal_train_eval_id_pos,normal_train_eval_id_nor),normal_train_eval_id_neg)
    simple_train_id_list = []
    for i in range(0,len(simple_train_id_list_pos)):
        simple_train_id_list.append(np.append(np.append(simple_train_id_list_pos[i],simple_train_id_list_nor[i]),simple_train_id_list_neg[i]))
    simple_eval_id_list = []
    for i in range(0,len(simple_eval_id_list_pos)):
        simple_eval_id_list.append(np.append(np.append(simple_eval_id_list_pos[i],simple_eval_id_list_nor[i]),simple_eval_id_list_neg[i]))

    ls2file(test_id, 'dongfan_data/data_kind_split/hyper-test-id.txt')
    ls2file(normal_train_train_id, 'dongfan_data/data_kind_split/hyper-normal-train-id.txt')
    ls2file(normal_train_eval_id, 'dongfan_data/data_kind_split/hyper-normal-eval-id.txt')
    count = 0
    for i in range(len(simple_train_id_list)):
        count+=1
        ls2file(simple_train_id_list[i],'dongfan_data/data_kind_split/hyper-train-id-'+str(count)+'.txt')
        ls2file(simple_eval_id_list[i],'dongfan_data/data_kind_split/hyper-eval-id-'+str(count)+'.txt')
        # print('train size: %d, eval size: %d' %
        #       (len(simple_train_id), len(simple_eval_id)))



def split_dataset(data_path='data_kind_split/hyper-multi-visit.pkl'):

    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')

    data = pd.read_pickle(data_path)
    sample_id = data['PERSON_INFO_ID'].unique()
    print(len(sample_id))

    random_number = [i for i in range(len(sample_id))]
    # 随机序号
    shuffle(random_number)

    # 总的训练+验证 训练+验证 80% 测试 20%
    train_eval_id = sample_id[random_number[:int(len(sample_id) * 4 / 5)]]
    # 总的测试
    test_id = sample_id[random_number[int(len(sample_id)*4/5):]]

    random_train_eval_number = [i for i in range(len(train_eval_id))]
    shuffle(random_train_eval_number)
    # 用于一般训练的数据 30%
    normal_train_id = train_eval_id[random_train_eval_number[:int(len(train_eval_id) * 3 / 10)]]
    random_normal_train_number = [i for i in range(len(normal_train_id))]
    shuffle(random_normal_train_number)
    normal_train_train_id = normal_train_id[random_normal_train_number[:int(len(normal_train_id) * 4 / 5)]]
    normal_train_eval_id = normal_train_id[random_normal_train_number[int(len(normal_train_id) * 4 / 5):]]
    ls2file(normal_train_train_id, '琼海市塔洋镇卫生院/data_7/hyper-normal-train-id.txt')
    ls2file(normal_train_eval_id, '琼海市塔洋镇卫生院/data_7/hyper-normal-eval-id.txt')
    print('normal train size: %d' % len(normal_train_train_id))
    print('normal eval size: %d' % len(normal_train_eval_id))


    # 用于增量学习的数据 70%
    inc_train_eval_id = train_eval_id[random_train_eval_number[int(len(train_eval_id) * 3 / 10):]]

    # set_test = set(test_id)


    ls2file(test_id, '琼海市塔洋镇卫生院/data_7/hyper-test-id.txt')
    # ls2file(train_id, 'zsw_hyper/data_7/hyper-train-id.txt')
    print('test size: %d' %len(test_id))
    # print('train size: %d' % len(train_id))

    #
    random_inc_train_eval_number =  [i for i in range(len(inc_train_eval_id))]
    shuffle(random_inc_train_eval_number)

    # 增量学习 分成7组训练+验证
    all_inc_train_eval_list = []
    for i in range(0,7):
        simple_train_eval_id = inc_train_eval_id[random_inc_train_eval_number[int(len(inc_train_eval_id) * i / 7):int(len(inc_train_eval_id) * (i+1) / 7)]]
        all_inc_train_eval_list.append(simple_train_eval_id)



    # 每单组再分为5份
    count = 0
    for simple_train_eval_id in all_inc_train_eval_list:
        count+=1
        random_simple_train_eval_number = [i for i in range(len(simple_train_eval_id))]
        shuffle(random_simple_train_eval_number)
        simple_train_id = simple_train_eval_id[random_simple_train_eval_number[:int(len(simple_train_eval_id) * 4 / 5)]]
        simple_eval_id = simple_train_eval_id[random_simple_train_eval_number[int(len(simple_train_eval_id)*4/5):]]
        ls2file(simple_train_id,'琼海市塔洋镇卫生院/data_7/hyper-train-id-'+str(count)+'.txt')
        ls2file(simple_eval_id,'琼海市塔洋镇卫生院/data_7/hyper-eval-id-'+str(count)+'.txt')
        print('train size: %d, eval size: %d' %
              (len(simple_train_id), len(simple_eval_id)))
        # set_test = set_test | set(simple_train_id)
        # set_test = set_test | set(simple_eval_id)

    # train_id = sample_id[random_number[:int(len(sample_id)*1/2)]]
    # eval_id = sample_id[random_number[int(
    #     len(sample_id)*1/2): int(len(sample_id)*2/3)]]
    # test_id = sample_id[random_number[int(len(sample_id)*2/3):]]



    # ls2file(train_id, 'zsw_hyper/hyper-train-id.txt')
    # ls2file(eval_id, 'zsw_hyper/hyper-eval-id.txt')
    # ls2file(test_id, 'zsw_hyper/hyper-test-id.txt')

    # print('train size: %d, eval size: %d, test size: %d' %
    #       (len(train_id), len(eval_id), len(test_id)))
# 2 8 交叉验证
def split_dataset_jiaocha(data_path='gonggong/hyper-multi-visit.pkl'):

    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')

    data = pd.read_pickle(data_path)
    sample_id = data['PERSON_INFO_ID'].unique()
    print(len(sample_id))

    random_number = [i for i in range(len(sample_id))]
    # 随机序号
    shuffle(random_number)

    # 分成五份 四份做测试 1份做训练+验证
    for i in range(0, 5):
        data_jiaocha_id = sample_id[random_number[int(len(sample_id) * i / 5):int(len(sample_id) * (i+1) / 5)]]
        ls2file(data_jiaocha_id, path+'data_jiaochayanzheng/hyper-data-'+str(i+1)+'.txt')
        random_train_eval_number = [i for i in range(len(data_jiaocha_id))]
        shuffle(random_train_eval_number)
        train_id = data_jiaocha_id[random_train_eval_number[:int(len(data_jiaocha_id) * 4 / 5)]]
        eval_id = data_jiaocha_id[random_train_eval_number[int(len(data_jiaocha_id) * 4 / 5):]]
        ls2file(train_id, path+'data_jiaochayanzheng/hyper-train-id-' + str(i+1) + '.txt')
        ls2file(eval_id, path+'data_jiaochayanzheng/hyper-eval-id-' + str(i+1) + '.txt')

    # # 总的训练+验证 训练+验证 20% 测试 80%
    # train_eval_id = sample_id[random_number[:int(len(sample_id) * 1 / 5)]]
    # # 总的测试
    # test_id = sample_id[random_number[int(len(sample_id)*1/5):]]
    #
    # random_train_eval_number = [i for i in range(len(train_eval_id))]
    # shuffle(random_train_eval_number)
    # # 用于一般训练的数据 30%
    # normal_train_id = train_eval_id[random_train_eval_number[:int(len(train_eval_id) * 3 / 10)]]
    # random_normal_train_number = [i for i in range(len(normal_train_id))]
    # shuffle(random_normal_train_number)
    # normal_train_train_id = normal_train_id[random_normal_train_number[:int(len(normal_train_id) * 4 / 5)]]
    # normal_train_eval_id = normal_train_id[random_normal_train_number[int(len(normal_train_id) * 4 / 5):]]
    # ls2file(normal_train_train_id, '琼海市塔洋镇卫生院/data_7/hyper-normal-train-id.txt')
    # ls2file(normal_train_eval_id, '琼海市塔洋镇卫生院/data_7/hyper-normal-eval-id.txt')
    # print('normal train size: %d' % len(normal_train_train_id))
    # print('normal eval size: %d' % len(normal_train_eval_id))
    #
    #
    # # 用于增量学习的数据 70%
    # inc_train_eval_id = train_eval_id[random_train_eval_number[int(len(train_eval_id) * 3 / 10):]]
    #
    # # set_test = set(test_id)
    #
    #
    # ls2file(test_id, '琼海市塔洋镇卫生院/data_7/hyper-test-id.txt')
    # # ls2file(train_id, 'zsw_hyper/data_7/hyper-train-id.txt')
    # print('test size: %d' %len(test_id))
    # # print('train size: %d' % len(train_id))
    #
    # #
    # random_inc_train_eval_number =  [i for i in range(len(inc_train_eval_id))]
    # shuffle(random_inc_train_eval_number)
    #
    # # 增量学习 分成7组训练+验证
    # all_inc_train_eval_list = []
    # for i in range(0,7):
    #     simple_train_eval_id = inc_train_eval_id[random_inc_train_eval_number[int(len(inc_train_eval_id) * i / 7):int(len(inc_train_eval_id) * (i+1) / 7)]]
    #     all_inc_train_eval_list.append(simple_train_eval_id)
    #
    #
    #
    # # 每单组再分为5份
    # count = 0
    # for simple_train_eval_id in all_inc_train_eval_list:
    #     count+=1
    #     random_simple_train_eval_number = [i for i in range(len(simple_train_eval_id))]
    #     shuffle(random_simple_train_eval_number)
    #     simple_train_id = simple_train_eval_id[random_simple_train_eval_number[:int(len(simple_train_eval_id) * 4 / 5)]]
    #     simple_eval_id = simple_train_eval_id[random_simple_train_eval_number[int(len(simple_train_eval_id)*4/5):]]
    #     ls2file(simple_train_id,'琼海市塔洋镇卫生院/data_7/hyper-train-id-'+str(count)+'.txt')
    #     ls2file(simple_eval_id,'琼海市塔洋镇卫生院/data_7/hyper-eval-id-'+str(count)+'.txt')
    #     print('train size: %d, eval size: %d' %
    #           (len(simple_train_id), len(simple_eval_id)))
        # set_test = set_test | set(simple_train_id)
        # set_test = set_test | set(simple_eval_id)

    # train_id = sample_id[random_number[:int(len(sample_id)*1/2)]]
    # eval_id = sample_id[random_number[int(
    #     len(sample_id)*1/2): int(len(sample_id)*2/3)]]
    # test_id = sample_id[random_number[int(len(sample_id)*2/3):]]



    # ls2file(train_id, 'zsw_hyper/hyper-train-id.txt')
    # ls2file(eval_id, 'zsw_hyper/hyper-eval-id.txt')
    # ls2file(test_id, 'zsw_hyper/hyper-test-id.txt')

    # print('train size: %d, eval size: %d, test size: %d' %
    #       (len(train_id), len(eval_id), len(test_id)))

#
def split_dataset_pretrain():

    data_path_single = 'gonggong/hyper-single-visit.pkl'
    data_path_multi = 'gonggong/hyper-multi-visit.pkl'
    def ls2file(list_data, file_name):
        with open(file_name, 'w') as fout:
            for item in list_data:
                fout.write(str(item) + '\n')

    data_single = pd.read_pickle(data_path_single)
    data_multi = pd.read_pickle(data_path_multi)

    sample_id_single = data_single['PERSON_INFO_ID'].unique()
    sample_id_multi = data_multi['PERSON_INFO_ID'].unique()

    # print(len(sample_id))

    random_number_single = [i for i in range(len(sample_id_single))]
    random_number_multi = [i for i in range(len(sample_id_multi))]

    # 随机序号
    shuffle(random_number_single)
    shuffle(random_number_multi)

    train_single_id = random_number_single;
    ls2file(train_single_id, path + 'hyper-train-id.txt')

    eval_multi_id = random_number_multi[:int(len(random_number_multi) * 1 / 5)]
    test_multi_id = random_number_multi[int(len(random_number_multi) * 1 / 5):]

    ls2file(eval_multi_id, path + 'hyper-eval-id.txt')
    ls2file(test_multi_id, path + 'hyper-test-id.txt')





def main():
    singledata, age1,body_tem1, pulse_rate1, breathing_rate1, lsbp1, ldbp1, rsbp1, rdbp1, height1, weight1, waist1, bmi1, exercise_freq1, smoking_status1, \
    drinking_freq1, heart_rate1, tcho1,tg1,ldlc1,hdlc1,is_hyper1 = process_file(single_file)
    #
    multidata, age2,body_tem2, pulse_rate2, breathing_rate2, lsbp2, ldbp2, rsbp2, rdbp2, height2, weight2, waist2, bmi2, exercise_freq2, smoking_status2, \
    drinking_freq2, heart_rate2, tcho2,tg2,ldlc2,hdlc2,is_hyper2 = process_file(multi_last_file)

    unique_age = age1 | age2
    unique_body_tem = body_tem1 | body_tem2
    unique_pulse_rate = pulse_rate1 | pulse_rate2
    unique_breathing_rate = breathing_rate1 | breathing_rate2
    unique_lsbp = lsbp1 | lsbp2
    unique_ldbp = ldbp1 | ldbp2
    unique_rsbp = rsbp1 | rsbp2
    unique_rdbp = rdbp1 | rdbp2
    unique_height = height1 | height2
    unique_weight = weight1 | weight2
    unique_waist = waist1 | waist2
    unique_bmi = bmi1 | bmi2
    unique_exercise_freq = exercise_freq1 | exercise_freq2
    unique_smoking_status = smoking_status1 | smoking_status2
    unique_drinking_freq = drinking_freq1 | drinking_freq2
    unique_heart_rate = heart_rate1 | heart_rate2
    unique_tcho = tcho1 | tcho2
    unique_tg = tg1 | tg2
    unique_ldlc = ldlc1 | ldlc2
    unique_hdlc = hdlc1 | hdlc2
    unique_is_hyper = is_hyper1 | is_hyper2
    #
    #
    # path = "wmz/非均衡数据/"
    print("unique_age ",type(unique_age))
    write_txt(path+'age-vocab.txt', unique_age)
    write_txt(path+'age-vocab-multi.txt', age2)
    write_txt(path+'body-tem-vocab.txt',unique_body_tem)
    write_txt(path+'body-tem-vocab-multi.txt', body_tem2)
    write_txt(path+'pulse_rate-vocab.txt', unique_pulse_rate)
    write_txt(path+'pulse_rate-vocab-multi.txt', pulse_rate2)
    write_txt(path+'breathing_rate-vocab.txt', unique_breathing_rate)
    write_txt(path+'breathing_rate-vocab-multi.txt', breathing_rate2)
    write_txt(path+'lsbp-vocab.txt', unique_lsbp)
    write_txt(path+'lsbp-vocab-multi.txt', lsbp2)
    write_txt(path+'ldbp-vocab.txt', unique_ldbp)
    write_txt(path+'ldbp-vocab-multi.txt', ldbp2)
    write_txt(path+'rsbp-vocab.txt', unique_rsbp)
    write_txt(path+'rsbp-vocab-multi.txt', rsbp2)
    write_txt(path+'rdbp-vocab.txt', unique_rdbp)
    write_txt(path+'rdbp-vocab-multi.txt', rdbp2)
    write_txt(path+'height-vocab.txt', unique_height)
    write_txt(path+'height-vocab-multi.txt', height2)
    write_txt(path+'weight-vocab.txt', unique_weight)
    write_txt(path+'weight-vocab-multi.txt', weight2)
    write_txt(path+'waist-vocab.txt', unique_waist)
    write_txt(path+'waist-vocab-multi.txt', waist2)
    write_txt(path+'bmi-vocab.txt', unique_bmi)
    write_txt(path+'bmi-vocab-multi.txt', bmi2)
    write_txt(path+'exercise_freq-vocab.txt', unique_exercise_freq)
    write_txt(path+'exercise_freq-vocab-multi.txt', exercise_freq2)
    write_txt(path+'smoking_status-vocab.txt', unique_smoking_status)
    write_txt(path+'smoking_status-vocab-multi.txt', smoking_status2)
    write_txt(path+'drinking_freq-vocab.txt', unique_drinking_freq)
    write_txt(path+'drinking_freq-vocab-multi.txt', drinking_freq2)
    write_txt(path+'heart_rate-vocab.txt', unique_heart_rate)
    write_txt(path+'heart_rate-vocab-multi.txt', heart_rate2)
    write_txt(path+'tcho-vocab.txt', unique_tcho)
    write_txt(path+'tcho-vocab-multi.txt', tcho2)
    write_txt(path+'tg-vocab.txt', unique_tg)
    write_txt(path+'tg-vocab-multi.txt', tg2)
    write_txt(path+'ldlc-vocab.txt', unique_ldlc)
    write_txt(path+'ldlc-vocab-multi.txt', ldlc2)
    write_txt(path+'hdlc-vocab.txt', unique_hdlc)
    write_txt(path+'hdlc-vocab-multi.txt', hdlc2)

    write_txt(path+'is_hyper-vocab.txt', unique_is_hyper)
    write_txt(path+'is_hyper-vocab-multi.txt', is_hyper2)

    # save data
    singledata.to_pickle(path+'hyper-single-visit.pkl')
    multidata.to_pickle(path+'hyper-multi-visit.pkl')

    split_dataset_jiaocha()

    split_dataset_pretrain()

main()