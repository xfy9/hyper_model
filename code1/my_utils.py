import functools
import math
import time
import torch
import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import math
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)



def log_func_time(func):
    @functools.wraps(func)
    def wrapper(*args,**kw):
        logging.info("Begin running : %s" % func.__name__)
        old_time = time.time()
        result = func(*args,**kw)
        logging.info("%s spent for %.4f s!" % (func.__name__,time.time()-old_time))
        logging.info("End running: %s \n",func.__name__)
        return result
    return wrapper


@log_func_time
def load_public_data(params):
    # load poi cat dict
    with open(params.poi_cat_dict_path, "rb") as f:
        poi_cat_dict = pickle.load(f)
    logging.info("Loading poi category dict done!")

    # load poi loc dict
    with open(params.poi_loc_dict_path, "rb") as f:
        poi_loc_dict = pickle.load(f)
    logging.info("Loading poi location dict done!")

    # load poi dist mat
    with open(params.poi_dist_mat_path, "rb") as f:
        poi_dist_mat = pickle.load(f)
    logging.info("Loading poi distance mat done!")

    # load cat sim mat
    with open(params.cat_sim_mat_path, "rb") as f:
        cat_sim_mat = pickle.load(f)
    logging.info("Loading poi category similarity done!")

    #load user embedding initialization
    with open(params.user_path,"rb") as f:
        s_u = pickle.load(f)
    logging.info("Loading initial user profiling done!")

    #load spatial KG embedding initialization
    with open(params.s_KG_path,"rb") as f:
        s_KG = pickle.load(f)
    logging.info("Loading intial spatial KG embedding done!")

    return poi_dist_mat, cat_sim_mat, poi_cat_dict, poi_loc_dict, s_u, s_KG

@log_func_time
def load_train_data(params):
    #load POI sequence for training
    with open(params.poi_list_train_path,"rb") as f:
        poi_list_train = pickle.load(f)
    logging.info("Loading poi sequence training done!")

    #load user list for training
    with open(params.user_list_train_path,"rb") as f:
        user_list_train = pickle.load(f)
    logging.info("Loading user list training done!")

    #load temporal context for training
    with open(params.temporal_train_path,"rb") as f:
        temporal_context_train = pickle.load(f)
    logging.info("Loading temporal context training done!")

    user_list_train = torch.tensor(np.array(user_list_train))
    poi_list_train = torch.tensor(np.array(poi_list_train))
    temporal_context_train = torch.tensor(np.array(temporal_context_train))


    return poi_list_train, user_list_train, temporal_context_train

@log_func_time
def load_test_data(params):
    # load POI sequence for testing
    with open(params.poi_list_test_path,"rb") as f:
        poi_list_test = pickle.load(f)
    logging.info("Loading poi sequence training done!")

    # load user list for testing
    with open(params.user_list_test_path,"rb") as f:
        user_list_test = pickle.load(f)
    logging.info("Loading user list training done!")

    # load temporal context for testing
    with open(params.temporal_test_path,"rb") as f:
        temporal_context_test = pickle.load(f)
    logging.info("Loading temporal context training done!")

    user_list_test = torch.tensor(np.array(user_list_test))
    poi_list_test = torch.tensor(np.array(poi_list_test))
    temporal_context_test = torch.tensor(np.array(temporal_context_test))

    return poi_list_test,user_list_test, temporal_context_test


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def reward1():
    return 0;

# def reward(lambda_l, lambda_c, lambda_p, predict_POI_index, real_POI_index, POI_cat_dict, POI_loc_dict, poi_dist_mat, cat_sim_mat):
def reward(predict_hyper, real_hyper):
    if predict_hyper==real_hyper: return 1;
    else: return 0;

def reward2(is_hyper_labels,predict_hyper,sig_action_value):
    r1=r2=0
    if is_hyper_labels[1]==predict_hyper:
        r1=0.5
    if predict_hyper==0:
        if sig_action_value[0][1]>0.5:
            r2-=(sig_action_value[0][1]-0.5)*2
        else :
            r2+= 1-(abs(sig_action_value[0][0]+sig_action_value[0][1]-1))

    if predict_hyper==1:
        if sig_action_value[0][0]>0.5:
            r2 -=(sig_action_value[0][0] - 0.5)*2
        else:
            r2 += 1 -(abs(sig_action_value[0][0] + sig_action_value[0][1] - 1))


    print("reward2:::  ","   r1: ",r1,"   r2: ",r2,"   r1+0.5*r2: ",r1+0.5*r2)

    return r1+0.5*r2

def reward3(action,action_value,is_hyper_label):
    # if action==is_hyper_label[1]:
    #     return 1;
    # else:
    #     return 0;
    r = 0
    if action!=is_hyper_label[1]:
        r-= 1
        return r
    else:
        r+=1

    # if(action==is_hyper_label[1]):
    #     r+=0.3
    left = action_value[0][0]
    right = action_value[0][1]
    if left ==is_hyper_label[0] or right==is_hyper_label[1]:
        r-= 1
        return r
    # 欧几里得距离
    # dis = math.sqrt((left-is_hyper_label[0])*(left-is_hyper_label[0])+(right-is_hyper_label[1])*(right-is_hyper_label[1]))
    dis = math.sqrt((left - is_hyper_label[0]) * (left - is_hyper_label[0]) + (right - is_hyper_label[1]) * (
                right - is_hyper_label[1]))
    # 曼哈顿距离
    # dis = (abs(left - is_hyper_label[0]) + abs(right - is_hyper_label[1])+0.01).cpu()
    #print("dis ",dis)

    # if(is_hyper_label[1]==0):
    #     r+=0.5*left;
    # else:
    #     r+=0.5*right;\
    if dis==0:
        #r+=0
        dis=0
    else:
        dis = sigmoid(1/dis)
        #dis = 1 / dis
        #print("dis  ",dis)
        r+=dis
    #print("r  ",r)

    return r;



def evaluate_model(y, y_pred, POI_cat_dict, POI_loc_dict, dist_mat, sim_mat):

    y_cat_true = list(map(lambda x: POI_cat_dict[x], y))
    y_cat_pred = list(map(lambda x: POI_cat_dict[x], y_pred))

    cat_df = pd.DataFrame({'true':y_cat_true, 'pred':y_cat_pred})
    cat_df['left'] = cat_df.apply(lambda x: min(x['true'], x['pred']), axis=1)
    cat_df['right'] = cat_df.apply(lambda x: max(x['true'], x['pred']), axis=1)
    cat_sim_all = cat_df.apply(lambda x: sim_mat[x['left'], x['right']], axis=1)

    y_dist_true = list(map(lambda x: POI_loc_dict[x], y))
    y_dist_pred = list(map(lambda x: POI_loc_dict[x], y_pred))
    dist_df = pd.DataFrame({'true':y_dist_true, 'pred':y_dist_pred})
    dist_df['left'] = dist_df.apply(lambda x: min(x['true'], x['pred']), axis=1)
    dist_df['right'] = dist_df.apply(lambda x: max(x['true'], x['pred']), axis=1)
    distance = dist_df.apply(lambda x: dist_mat[x['left'], x['right']], axis=1)



    cat_precision = precision_score(y_cat_true, y_cat_pred, average='weighted')
    cat_recall = recall_score(y_cat_true, y_cat_pred, average='weighted')
    avg_dist = sum(distance) / len(distance)
    avg_sim = sum(cat_sim_all) / len(cat_sim_all)

    return cat_precision, cat_recall, avg_sim, avg_dist


def write_txt(file_name,content):
    with open(file_name, "a",encoding='utf-8') as file:
        file.write(content + "\n")



# 高血压预警 结果转为一维向量
def change_narray_tensor(now_rx_y_preds):
    result_tensor = torch.empty(0)
    for i in range(len(now_rx_y_preds)):
        for j in range(len(now_rx_y_preds[i])):
            tmp_tensor = torch.Tensor(now_rx_y_preds[i][j])
            result_tensor = torch.cat([result_tensor, tmp_tensor], 0)

    return result_tensor

### 查看高血压数据列表
def see_group_is_hyper_label(temp_train_list,count):
    for train_input in temp_train_list:
        input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
        lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
        waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
        drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = train_input

        input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
        waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
            input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
            rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
            smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                dim=0)
        # print(is_hyper_labels)
        group_label_list = []
        for i in range(is_hyper_labels.size(0)):
            label_list = []
            label_list.append(int(is_hyper_labels[i][0]))
            label_list.append(int(is_hyper_labels[i][1]))
            # print(int(is_hyper_labels[i][0]), "    ", int(is_hyper_labels[i][1]))
            group_label_list.append(label_list)
        print(str(group_label_list))
        write_txt('./data_see/data_train_see_'+str(count)+'.txt',str(group_label_list))


