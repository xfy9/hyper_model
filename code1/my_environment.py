import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available()
                                and not False else "cpu")

from torch import nn




class Environment(torch.nn.Module):
    """
    Implementation for environment to update states
    """
    def __init__(self, n_state,data_shape,model_hyper,config,tokenizer):
        '''
        Initialization
        :param n_h: int, dimension of weights/state
        '''
        super(Environment, self).__init__()
        self.n_state = n_state
        self.data_shape = data_shape


        self.model_hyper = model_hyper
        # self.n_t_1 = n_t_1
        # self.n_t_2 = n_t_2

        # torch.nn.Parameter 可训练参数
        # torch.zeros 全为0
        # self.W_u = torch.nn.Parameter(data=torch.zeros((self.n_state, 1)), requires_grad=True)
        self.W_u = torch.nn.Parameter(data=torch.zeros((self.data_shape, self.n_state)), requires_grad=True)
        #self.W_u = self.W_u.detach()
        print("requires_grad ",self.W_u.requires_grad)



        # 随机生成
        self.W_u.data.uniform_(-1, 1)

        #高血压预警
        self.cls_predict = nn.Sequential(nn.Linear(40 * config.hidden_size, 2 * config.hidden_size), nn.ReLU(),
                                 nn.Linear(2 * config.hidden_size, len(tokenizer.is_hyper_voc_multi.word2idx)))

        self.cls_predict.to(device)

    def forward(self,inputs, age_labels=None, body_tem_labels=None, pulse_rate_labels=None, breathing_rate_labels=None,
                lsbp_labels=None, ldbp_labels=None, rsbp_labels=None, rdbp_labels=None,
                height_labels=None, weight_labels=None, waist_labels=None, bmi_labels=None,
                exercise_freq_labels=None, smoking_status_labels=None, drinking_freq_labels=None,
                heart_rate_labels=None, tcho_labels=None, tg_labels=None, ldlc_labels=None, hdlc_labels=None,
                is_hyper_labels=None, epoch=None,now_state=None,limit_num=None):
        bert_predict_tensor,data_len = self.model_hyper(inputs, age_labels=age_labels,body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                                        lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                                        height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                                        exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                                        heart_rate_labels=heart_rate_labels,tcho_labels=tcho_labels,tg_labels=tg_labels,ldlc_labels=ldlc_labels,hdlc_labels=hdlc_labels,is_hyper_labels=is_hyper_labels,epoch=None,
                                                        limit_num=limit_num)


        # print('env bert_predict_tensor  ', bert_predict_tensor[0].shape)

        #state = torch.mm(bert_predict_tensor.cpu(),self.W_u)
        state = bert_predict_tensor.cpu()

        return state,data_len
