
# from dqn import DQN
# from utils import reward,evaluate_model
import numpy as np
import logging
import torch
from my_environment import Environment
from my_dqn import DQN
from my_utils import reward,reward1,reward2,reward3,change_narray_tensor
from tqdm import tqdm, trange
from utils import metric_report, t2n, get_n_params
from torch.optim import Adam,SGD
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
import torch.nn.functional as F
import my_utils
from early_stopping import EarlyStopping
from datetime import datetime
import os


class RLKGModel(object):

    # def __init__(self,poi_info,user_KG,params):
    def __init__(self,params,model_env,tokenizer,device,config):
        # 地点信息
        # self.poi_info = poi_info
        # 地点知识图谱
        # self.user_KG = user_KG
        self.visit_counter = 0
        # self.user
        self.ll = params.ll
        self.lc = params.lc
        self.lp = params.lp
        # self.poi_cat_dict = poi_info.poi_cat_dict
        # self.poi_loc_dict = poi_info.poi_loc_dict
        # self.poi_dist_mat = poi_info.poi_dist_mat
        # self.cat_sim_mat  = poi_info.cat_sim_mat

        # self.n_state = 200
        self.n_state = 12000
        self.env_nt_1 = 2330
        self.env_nt_2 = 2
        self.data_shape = 12000
        self.memory_capacity = 10
        # self.lr = 1e-3
        self.lr = 5e-4
        self.epsilon=0.9
        self.batch_size=1
        self.gamma=0.5  #
        self.target_replace_iter =5
        self.num_hyper=2

        # self.s_u =torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)
        # self.s_u.data.uniform_(-1, 1)


        #高血压预警
        self.config = config
        # 内存容量 到达某个状态就更新
        # self.memory_capacity = params.memory_capacity
        #
        #
        self.state =torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)
        self.state.data.uniform_(-1, 1)

        # 高血压预警模型
        self.model_env=model_env;
        self.tokenizer = tokenizer

        self.environment = Environment(self.n_state,
                                       self.data_shape,
                                       self.model_env,
                                       self.config,
                                       self.tokenizer)

        self.device = device



        # self
        #
        self.dqn = DQN(self.environment,
                       self.n_state,
                       self.num_hyper,
                       self.memory_capacity,
                       self.lr,
                       self.epsilon,
                       self.batch_size,
                       self.gamma,
                       self.target_replace_iter,
                       self.tokenizer,
                       self.device,
                       mode='lr')

        # self.predict_POI_index = np.random.randint(user_KG.s_KG.num_POI)
        self.predict_hyper = np.random.randint(0,2)
        self.predict_hyper1 = np.random.randint(0, 2)
        self.r = reward1()
        self.optimizer3 = SGD(self.model_env.parameters(), lr=3.5e-4)

        #加载高血压预警模型



    # 在强化学习框架中加入高血压预警模型
    def fit2(self,train_dataloader,device):
        print("fit2")
        train_loss = 0
        all_count=0
        for train_input in tqdm(train_dataloader, desc="Training"):
            train_input = tuple(t.to(device) for t in train_input)
            pre_state = torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)
            pre_state1 = torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)
            self.predict_hyper = np.random.randint(0, 2)
            self.predict_hyper1 = np.random.randint(0, 2)
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

            preds = []
            preds_logists = []

            #print('len',is_hyper_labels)
            for i in range(is_hyper_labels.size(0)):
                all_count += 1
                now_state, data_len = self.environment(input_ids, age_labels=age_labels,
                                                             body_tem_labels=body_tem_labels,
                                                             pulse_rate_labels=pulse_rate_labels,
                                                             breathing_rate_labels=breathing_rate_labels,
                                                             lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                             rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                             height_labels=height_labels, weight_labels=weight_labels,
                                                             waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                             exercise_freq_labels=exercise_freq_labels,
                                                             smoking_status_labels=smoking_status_labels,
                                                             drinking_freq_labels=drinking_freq_labels,
                                                             heart_rate_labels=heart_rate_labels,
                                                             tcho_labels=tcho_labels,
                                                             tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                             hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                             limit_num=i + 1)

                # print("env is_hyper_logits", is_hyper_logits)
                # #print('state ',self.state[0][0])
                # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                # # self.state[0][i*2]=is_hyper_logits[i][0];
                # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                # print('now_state ', now_state)
                now_state = now_state.cpu()
                #print("now_state ", now_state)
                # print("pre_state", pre_state)

                # real_state = is_hyper_labels[0:i+1,:].clone()
                # print('real_state ',real_state)
                # real_state = real_state.reshape(-1).cpu().unsqueeze(0)
                # print('type ', real_state)
                if i>0 and is_hyper_labels[i][0]!=is_hyper_labels[i-1][0]:
                #if True:
                    self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                    #preds.append(self.predict_hyper[0])
                    preds_logists.append(torch.Tensor(hyper_value))
                    self.r = reward3(action=self.predict_hyper, action_value=hyper_value,
                                     is_hyper_label=is_hyper_labels[i])
                    self.dqn.store_transition(now_state, self.predict_hyper, self.r, pre_state)

                    pre_state = now_state

                    # print('self.predict_hyper ', self.predict_hyper, 'hyper_value ', hyper_value, 'is_hyper_labels',
                    #       is_hyper_labels[i])


                else :
                    self.predict_hyper1, hyper_value1 = self.dqn.choose_action(now_state)
                    preds_logists.append(torch.Tensor(hyper_value1))
                    self.r = reward3(action=self.predict_hyper1, action_value=hyper_value1,
                                     is_hyper_label=is_hyper_labels[i])
                    self.dqn.store_transition1(now_state, self.predict_hyper1, self.r, pre_state)

                    pre_state = now_state

                    # print('self.predict_hyper1 ', self.predict_hyper1, 'hyper_value1 ', hyper_value1, 'is_hyper_labels',
                    #       is_hyper_labels[i])





                if self.dqn.memory_counter > self.memory_capacity:
                    loss = self.dqn.learn(flag=1)
                if self.dqn.memory_counter1 > 1:
                    loss = self.dqn.learn(flag=0)
            preds_logists = torch.cat(preds_logists, dim=0).to(device)
            # print("is_hyper_labels  type  ",is_hyper_labels)
            # print("preds_logists  type  ", preds_logists)
            loss = F.binary_cross_entropy_with_logits(preds_logists, is_hyper_labels)
            train_loss += loss.item()
        return train_loss/all_count;

                # pre_state = now_state
                # self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                # print('self.predict_hyper ', self.predict_hyper, 'hyper_value ', hyper_value, 'is_hyper_labels',
                #       is_hyper_labels[i])
                #
                # self.r = reward3(action=self.predict_hyper, action_value=hyper_value, is_hyper_label=is_hyper_labels[i])





    # 对模型进行优化
    def fit3(self,test_input):
        pre_state = torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)
        # pre_state.data.uniform_(-1, 1)
        print("fit3")

        input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
        lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
        waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
        drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = test_input

        input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
        waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
            input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
            rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
            smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                dim=0)

        is_hyper_logits_clone, data_len = self.environment(input_ids, age_labels=age_labels,
                                                           body_tem_labels=body_tem_labels,
                                                           pulse_rate_labels=pulse_rate_labels,
                                                           breathing_rate_labels=breathing_rate_labels,
                                                           lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                           rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                           height_labels=height_labels, weight_labels=weight_labels,
                                                           waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                           exercise_freq_labels=exercise_freq_labels,
                                                           smoking_status_labels=smoking_status_labels,
                                                           drinking_freq_labels=drinking_freq_labels,
                                                           heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                           tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                           hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                           limit_num=9999)
        print("is_hyper_logits_clone ",is_hyper_logits_clone,"  is_hyper_labels  ",is_hyper_labels)
        loss = F.binary_cross_entropy_with_logits(is_hyper_logits_clone, is_hyper_labels)

        loss.backward()
        self.optimizer3.step()
        self.optimizer3.zero_grad()

        for i in range(is_hyper_labels.size(0)):
            is_hyper_logits,data_len = self.environment(input_ids, age_labels=age_labels,
                                                   body_tem_labels=body_tem_labels,
                                                   pulse_rate_labels=pulse_rate_labels,
                                                   breathing_rate_labels=breathing_rate_labels,
                                                   lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                   rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                   height_labels=height_labels, weight_labels=weight_labels,
                                                   waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                   exercise_freq_labels=exercise_freq_labels,
                                                   smoking_status_labels=smoking_status_labels,
                                                   drinking_freq_labels=drinking_freq_labels,
                                                   heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                   tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                   hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,limit_num=i+1)




            # print("env is_hyper_logits", is_hyper_logits)
            # #print('state ',self.state[0][0])
            # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
            # # self.state[0][i*2]=is_hyper_logits[i][0];
            # # self.state[0][i*2+1]=is_hyper_logits[i][1];
            now_state = is_hyper_logits.reshape(-1).cpu()
            # print('now_state ', now_state)
            rest_state =torch.zeros(self.n_state-now_state.shape[0])
            now_state = torch.cat((now_state,rest_state),0).unsqueeze(0)

            print("now_state ", now_state)
            # print("pre_state", pre_state)

            # real_state = is_hyper_labels[0:i+1,:].clone()
            # print('real_state ',real_state)
            # real_state = real_state.reshape(-1).cpu().unsqueeze(0)
            # print('type ', real_state)



            self.dqn.store_transition(now_state,self.predict_hyper,self.r,pre_state)

            if self.dqn.memory_counter > self.memory_capacity:
                self.dqn.learn()

            pre_state = now_state
            self.predict_hyper,hyper_value = self.dqn.choose_action(now_state)
            print('self.predict_hyper ',self.predict_hyper,'hyper_value ',hyper_value,'is_hyper_labels',is_hyper_labels[i])

            self.r = reward3(action=self.predict_hyper,action_value=hyper_value,is_hyper_label=is_hyper_labels[i])

    def fit4(self, train_dataloader,device):

        # pre_state.data.uniform_(-1, 1)
        y_preds = []
        y_trues = []
        print("fit4")
        for train_input in tqdm(train_dataloader, desc="Training"):
            train_input = tuple(t.to(device) for t in train_input)
            pre_state = torch.nn.Parameter(data=torch.zeros((1, self.n_state)), requires_grad=True)
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


            #print('is_hyper_labels.size(0)  ',is_hyper_labels.size(0))
            preds = []
            preds_logists = []
            for i in range(is_hyper_labels.size(0)):
                is_hyper_logits, data_len = self.environment(input_ids, age_labels=age_labels,
                                                             body_tem_labels=body_tem_labels,
                                                             pulse_rate_labels=pulse_rate_labels,
                                                             breathing_rate_labels=breathing_rate_labels,
                                                             lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                             rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                             height_labels=height_labels, weight_labels=weight_labels,
                                                             waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                             exercise_freq_labels=exercise_freq_labels,
                                                             smoking_status_labels=smoking_status_labels,
                                                             drinking_freq_labels=drinking_freq_labels,
                                                             heart_rate_labels=heart_rate_labels,
                                                             tcho_labels=tcho_labels,
                                                             tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                             hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                             limit_num=i + 1)

                # print("env is_hyper_logits", is_hyper_logits)
                # #print('state ',self.state[0][0])
                # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                # # self.state[0][i*2]=is_hyper_logits[i][0];
                # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                now_state = is_hyper_logits.reshape(-1).cpu()
                # print('now_state ', now_state)
                rest_state = torch.zeros(self.n_state - now_state.shape[0])
                now_state = torch.cat((now_state, rest_state), 0).unsqueeze(0)

                # print("now_state ", now_state)
                # print("pre_state", pre_state)

                # real_state = is_hyper_labels[0:i+1,:].clone()
                # print('real_state ',real_state)
                # real_state = real_state.reshape(-1).cpu().unsqueeze(0)
                # print('type ', real_state)

                self.dqn.store_transition(now_state, self.predict_hyper, self.r, pre_state)

                if self.dqn.memory_counter > self.memory_capacity:
                    self.dqn.learn()

                pre_state = now_state
                self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                # print('self.predict_hyper ', self.predict_hyper, 'hyper_value ', hyper_value, 'is_hyper_labels',
                #       is_hyper_labels[i])

                self.r = reward3(action=self.predict_hyper, action_value=hyper_value, is_hyper_label=is_hyper_labels[i])
                preds.append(hyper_value[0])
                preds_logists.append(torch.Tensor(hyper_value))

            y_preds.append(np.array(preds))
            y_trues.append(t2n(is_hyper_labels))


        rx_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                         0.3)
        is_hyper_logits_clone, data_len = self.environment(input_ids, age_labels=age_labels,
                                                           body_tem_labels=body_tem_labels,
                                                           pulse_rate_labels=pulse_rate_labels,
                                                           breathing_rate_labels=breathing_rate_labels,
                                                           lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                           rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                           height_labels=height_labels, weight_labels=weight_labels,
                                                           waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                           exercise_freq_labels=exercise_freq_labels,
                                                           smoking_status_labels=smoking_status_labels,
                                                           drinking_freq_labels=drinking_freq_labels,
                                                           heart_rate_labels=heart_rate_labels,
                                                           tcho_labels=tcho_labels,
                                                           tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                           hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                           limit_num=9999)
        # print("is_hyper_logits_clone ", is_hyper_logits_clone, "  is_hyper_labels  ", is_hyper_labels)
        loss = F.binary_cross_entropy_with_logits(is_hyper_logits_clone, is_hyper_labels)

        loss.backward()
        self.optimizer3.step()
        self.optimizer3.zero_grad()
        return rx_acc_container













    def evaluate2(self,eval_dataloader,device,temp_count):
        y_preds = []
        y_trues = []
        eval_loss = 0
        count=0;
        for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
            if(count>=temp_count):break
            count+=1
            eval_input = tuple(t.to(device) for t in eval_input)

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
            lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
            drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = eval_input

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
                input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
                smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                    dim=0)
            preds = []
            preds_logists = []
            #print('is_hyper_labels.size(0)  ', is_hyper_labels.size(0))
            for i in range(is_hyper_labels.size(0)):
                is_hyper_logits, data_len = self.environment(input_ids, age_labels=age_labels,
                                                             body_tem_labels=body_tem_labels,
                                                             pulse_rate_labels=pulse_rate_labels,
                                                             breathing_rate_labels=breathing_rate_labels,
                                                             lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                             rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                             height_labels=height_labels, weight_labels=weight_labels,
                                                             waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                             exercise_freq_labels=exercise_freq_labels,
                                                             smoking_status_labels=smoking_status_labels,
                                                             drinking_freq_labels=drinking_freq_labels,
                                                             heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                             tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                             hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                             limit_num=i + 1)

                # print("env is_hyper_logits", is_hyper_logits)
                # #print('state ',self.state[0][0])
                # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                # # self.state[0][i*2]=is_hyper_logits[i][0];
                # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                now_state = is_hyper_logits.reshape(-1).cpu()
                #print('now_state ', now_state)
                rest_state = torch.zeros(self.n_state - now_state.shape[0])
                now_state = torch.cat((now_state, rest_state), 0).unsqueeze(0)
                # print("now_state ", now_state)
                # print("pre_state", pre_state)
                self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                #print('hyper_value' ,hyper_value)
                preds.append(hyper_value[0])
                preds_logists.append(torch.Tensor(hyper_value))
            y_preds.append(np.array(preds))
            y_trues.append(t2n(is_hyper_labels))


            preds_logists = torch.cat(preds_logists, dim=0).to(device)
            #print("is_hyper_labels  type  ",is_hyper_labels)
            #print("preds_logists  type  ", preds_logists)
            loss = F.binary_cross_entropy_with_logits(preds_logists, is_hyper_labels)
            eval_loss+=loss.item()



        rx_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                         0.3)
        #my_utils.write_txt('./result_train_guding10000_100.txt', str(rx_acc_container))
        print(rx_acc_container)
        return eval_loss / count


    def evaluate3(self,eval_dataloader,device,temp_count):
        y_preds = []
        y_trues = []
        eval_loss = 0
        count=0;
        for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
            # if(count>=temp_count):break

            eval_input = tuple(t.to(device) for t in eval_input)

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
            lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
            drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = eval_input

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
                input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
                smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                    dim=0)
            preds = []
            preds_logists = []
            #print('is_hyper_labels.size(0)  ', is_hyper_labels.size(0))
            for i in range(is_hyper_labels.size(0)):
                count += 1
                now_state, data_len = self.environment(input_ids, age_labels=age_labels,
                                                             body_tem_labels=body_tem_labels,
                                                             pulse_rate_labels=pulse_rate_labels,
                                                             breathing_rate_labels=breathing_rate_labels,
                                                             lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                             rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                             height_labels=height_labels, weight_labels=weight_labels,
                                                             waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                             exercise_freq_labels=exercise_freq_labels,
                                                             smoking_status_labels=smoking_status_labels,
                                                             drinking_freq_labels=drinking_freq_labels,
                                                             heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                             tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                             hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                             limit_num=i + 1)

                now_state = now_state.cpu()

                # print("env is_hyper_logits", is_hyper_logits)
                # #print('state ',self.state[0][0])
                # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                # # self.state[0][i*2]=is_hyper_logits[i][0];
                # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                # print("now_state ", now_state)
                # print("pre_state", pre_state)
                self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                #print('hyper_value' ,hyper_value)
                preds.append(hyper_value[0])
                preds_logists.append(torch.Tensor(hyper_value))
            y_preds.append(np.array(preds))
            y_trues.append(t2n(is_hyper_labels))


            preds_logists = torch.cat(preds_logists, dim=0).to(device)
            #print("is_hyper_labels  type  ",is_hyper_labels)
            #print("preds_logists  type  ", preds_logists)
            loss = F.binary_cross_entropy_with_logits(preds_logists, is_hyper_labels)
            eval_loss+=loss.item()



        rx_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                         0.5)
        #my_utils.write_txt('./result_train_guding10000_100.txt', str(rx_acc_container))
        print(rx_acc_container)
        return eval_loss / count


    def test(self,test_dataloader,device,temp_count,save_path):
        self.dqn.eval_net.load_state_dict(torch.load(save_path+"/best_network.pth"))
        y_preds = []
        y_trues = []
        eval_loss = 0
        count=0;
        for test_input in tqdm(test_dataloader, desc="Evaluating"):
            if(count>=temp_count):break
            count+=1
            test_input = tuple(t.to(device) for t in test_input)

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
            lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
            drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = test_input

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
                input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
                smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                    dim=0)
            preds = []
            preds_logists = []
            #print('is_hyper_labels.size(0)  ', is_hyper_labels.size(0))
            for i in range(is_hyper_labels.size(0)):
                is_hyper_logits, data_len = self.environment(input_ids, age_labels=age_labels,
                                                             body_tem_labels=body_tem_labels,
                                                             pulse_rate_labels=pulse_rate_labels,
                                                             breathing_rate_labels=breathing_rate_labels,
                                                             lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                             rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                             height_labels=height_labels, weight_labels=weight_labels,
                                                             waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                             exercise_freq_labels=exercise_freq_labels,
                                                             smoking_status_labels=smoking_status_labels,
                                                             drinking_freq_labels=drinking_freq_labels,
                                                             heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                             tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                             hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                             limit_num=i + 1)

                # print("env is_hyper_logits", is_hyper_logits)
                # #print('state ',self.state[0][0])
                # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                # # self.state[0][i*2]=is_hyper_logits[i][0];
                # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                now_state = is_hyper_logits.reshape(-1).cpu()
                #print('now_state ', now_state)
                rest_state = torch.zeros(self.n_state - now_state.shape[0])
                now_state = torch.cat((now_state, rest_state), 0).unsqueeze(0)
                # print("now_state ", now_state)
                # print("pre_state", pre_state)
                self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                #print('hyper_value' ,hyper_value)
                preds.append(hyper_value[0])
                preds_logists.append(torch.Tensor(hyper_value))
            y_preds.append(np.array(preds))
            y_trues.append(t2n(is_hyper_labels))


            preds_logists = torch.cat(preds_logists, dim=0).to(device)
            #print("is_hyper_labels  type  ",is_hyper_labels)
            #print("preds_logists  type  ", preds_logists)
            loss = F.binary_cross_entropy_with_logits(preds_logists, is_hyper_labels)
            eval_loss+=loss.item()



        rx_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                         0.3)
        curr_time = datetime.now()
        return rx_acc_container




    def test1(self,test_dataloader,device,temp_count,save_path):
        self.dqn.eval_net.load_state_dict(torch.load(save_path + "/best_network.pth"))
        y_preds = []
        y_trues = []
        eval_loss = 0
        count=0;
        for test_input in tqdm(test_dataloader, desc="Evaluating"):
            if(count>=temp_count):break
            count+=1
            test_input = tuple(t.to(device) for t in test_input)

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
            lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
            drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = test_input

            input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
            waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
                input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
                smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                    dim=0)
            preds = []
            preds_logists = []
            #print('is_hyper_labels.size(0)  ', is_hyper_labels.size(0))
            for i in range(is_hyper_labels.size(0)):
                now_state, data_len = self.environment(input_ids, age_labels=age_labels,
                                                             body_tem_labels=body_tem_labels,
                                                             pulse_rate_labels=pulse_rate_labels,
                                                             breathing_rate_labels=breathing_rate_labels,
                                                             lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                             rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                             height_labels=height_labels, weight_labels=weight_labels,
                                                             waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                             exercise_freq_labels=exercise_freq_labels,
                                                             smoking_status_labels=smoking_status_labels,
                                                             drinking_freq_labels=drinking_freq_labels,
                                                             heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                             tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                             hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                             limit_num=i + 1)

                # print("env is_hyper_logits", is_hyper_logits)
                # #print('state ',self.state[0][0])
                # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                # # self.state[0][i*2]=is_hyper_logits[i][0];
                # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                # print("now_state ", now_state)
                # print("pre_state", pre_state)
                now_state = now_state.cpu()
                self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                #print('hyper_value' ,hyper_value)
                preds.append(hyper_value[0])
                preds_logists.append(torch.Tensor(hyper_value))
            y_preds.append(np.array(preds))
            y_trues.append(t2n(is_hyper_labels))


            preds_logists = torch.cat(preds_logists, dim=0).to(device)
            #print("is_hyper_labels  type  ",is_hyper_labels)
            #print("preds_logists  type  ", preds_logists)
            loss = F.binary_cross_entropy_with_logits(preds_logists, is_hyper_labels)
            eval_loss+=loss.item()

        curr_time = datetime.now()  # 现在的时间
        today = str((curr_time.strftime("%Y-%m-%d")))  # 调用strftime方法就是对时间进行格式化


        rx_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                         0.5)
        # dir_name = './result/'+today
        # if not os.path.isdir(dir_name):
        #     os.makedirs(dir_name)
        # my_utils.write_txt(dir_name+'result_eval_bert.txt', str(rx_acc_container))
        # print(rx_acc_container)
        return rx_acc_container




    def test_jiaocha(self,test_dataloader_list,device,temp_count,save_path,i):
        self.dqn.eval_net.load_state_dict(torch.load(save_path + "/best_network.pth"))
        y_preds = []
        y_trues = []
        eval_loss = 0
        cishu=0
        print("i============",i)
        xi= i

        for test_dataloader in test_dataloader_list:
            cishu += 1
            if xi==cishu-1:
                continue
            print("cishu ", cishu)
            count = 0;
            for test_input in tqdm(test_dataloader, desc="Testing"):
                if(count>=temp_count):break
                count+=1
                test_input = tuple(t.to(device) for t in test_input)

                input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
                lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
                waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
                drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = test_input

                input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
                waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
                    input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                    rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
                    smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
                        dim=0)
                preds = []
                preds_logists = []
                #print('is_hyper_labels.size(0)  ', is_hyper_labels.size(0))
                for i in range(is_hyper_labels.size(0)):
                    now_state, data_len = self.environment(input_ids, age_labels=age_labels,
                                                                 body_tem_labels=body_tem_labels,
                                                                 pulse_rate_labels=pulse_rate_labels,
                                                                 breathing_rate_labels=breathing_rate_labels,
                                                                 lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,
                                                                 rsbp_labels=rsbp_labels, rdbp_labels=rdbp_labels,
                                                                 height_labels=height_labels, weight_labels=weight_labels,
                                                                 waist_labels=waist_labels, bmi_labels=bmi_labels,
                                                                 exercise_freq_labels=exercise_freq_labels,
                                                                 smoking_status_labels=smoking_status_labels,
                                                                 drinking_freq_labels=drinking_freq_labels,
                                                                 heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                                                 tg_labels=tg_labels, ldlc_labels=ldlc_labels,
                                                                 hdlc_labels=hdlc_labels, is_hyper_labels=is_hyper_labels,
                                                                 limit_num=i + 1)

                    # print("env is_hyper_logits", is_hyper_logits)
                    # #print('state ',self.state[0][0])
                    # print('is_hyper_logits[i][0] ',is_hyper_logits[i][0])
                    # # self.state[0][i*2]=is_hyper_logits[i][0];
                    # # self.state[0][i*2+1]=is_hyper_logits[i][1];
                    # print("now_state ", now_state)
                    # print("pre_state", pre_state)
                    now_state = now_state.cpu()
                    self.predict_hyper, hyper_value = self.dqn.choose_action(now_state)
                    #print('hyper_value' ,hyper_value)
                    preds.append(hyper_value[0])
                    preds_logists.append(torch.Tensor(hyper_value))
                y_preds.append(np.array(preds))
                y_trues.append(t2n(is_hyper_labels))


                preds_logists = torch.cat(preds_logists, dim=0).to(device)
                #print("is_hyper_labels  type  ",is_hyper_labels)
                #print("preds_logists  type  ", preds_logists)
                loss = F.binary_cross_entropy_with_logits(preds_logists, is_hyper_labels)
                eval_loss+=loss.item()

        curr_time = datetime.now()  # 现在的时间
        today = str((curr_time.strftime("%Y-%m-%d")))  # 调用strftime方法就是对时间进行格式化


        rx_acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                         0.5)
        return rx_acc_container


#
# for i in range(0,5):
#     j=0;
#     print("j================= ",i)
#     for k in range(0,5):
#         if j==i:
#             j+=1;
#             continue
#         else:
#             j+=1;
#         print(j)




