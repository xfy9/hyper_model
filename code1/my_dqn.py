import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import metric_report, t2n, get_n_params

import torch, random
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


class RBFN(nn.Module):
    def __init__(self, centers, n_out=10):
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.n_in = centers.size(1)
        self.num_centers = centers.size(0)
        #print("num_centers  ",self.num_centers)
        self.centers = nn.Parameter(centers)
        self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True)
        self.linear = nn.Linear(self.num_centers + self.n_in, self.n_out, bias=True)
        #self.linear = nn.Linear(self.num_centers, self.n_out, bias=True)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        #print("batches shape ",batches.shape)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        #print("A shape ", A.shape)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        #print("B shape ", B.shape)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False)))
        return C

    def forward(self, batches):
        radial_val = self.kernel_fun(batches)

        class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        return class_score

    def initialize_weights(self, ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)


class Net(torch.nn.Module):
    '''
        FC network

    '''

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        # self.fc1 = torch.nn.Linear(N_STATES, 100)
        self.centers  = torch.rand(100, N_STATES)
        self.fc1 = RBFN(self.centers, 100)
        #self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = torch.nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        # action_value shape [batch_size,N_ACTIONS]
        #print('action_value',action_value.shape)
        del x
        return action_value

class Net1(torch.nn.Module):
    '''
        FC network

    '''

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net1, self).__init__()
        # self.fc1 = torch.nn.Linear(N_STATES, 100)
        self.cls_predict = nn.Sequential(nn.Linear(N_STATES, 300), nn.ReLU(),
                                         nn.Linear(300, N_ACTIONS))

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        action_value = self.cls_predict(x)
        # action_value shape [batch_size,N_ACTIONS]
        #print('action_value',action_value.shape)
        del x
        return action_value


class DQN(object):

    def __init__(self, ENV, N_STATES, N_ACTIONS, MEMORY_CAPACITY, LR,
                 EPSILON, BATCH_SIZE, GAMMA,TARGET_REPLACE_ITER,tokenizer,device,mode='r'):
        '''
        Initialization
        :param ENV:
        :param N_STATES: dimensions of states 状态
        :param N_ACTIONS: number of actions 行动
        :param MEMORY_CAPACITY: 20
        :param LR: 学习率
        :param EPSILON: 0.9
        :param BATCH_SIZE: 6
        :param GAMMA: 0.9
        :param TARGET_REPLACE_ITER: 5
        '''
        self.N_STATES = N_STATES
        #print('self.N_STATES',self.N_STATES)
        #print('N_ACTIONS',N_ACTIONS)
        self.eval_net, self.target_net = Net1(self.N_STATES, N_ACTIONS), Net1(self.N_STATES, N_ACTIONS) # eval_net,

        # 高血压预警 eval target

        self.device = device

        self.env = ENV

        self.mode = mode

        self.N_ACTIONS = N_ACTIONS # number of actions
        # self.MEMORY_CAPACITY = MEMORY_CAPACITY # memory size for experience replay
        # self.MEMORY_CAPACITY1 = 1
        self.MEMORY_CAPACITY = MEMORY_CAPACITY # memory size for experience replay
        self.MEMORY_CAPACITY1 = 1
        self.EPSILON = EPSILON # epsilon greedy
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA # discount factor for TD error
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER # ???

        self.learn_step_counter = 0 # ???
        self.memory_counter = 0 # ???
        self.memory_counter1 = 0
        self.memory = np.zeros((MEMORY_CAPACITY, self.N_STATES * 2 + 2)) # initialization of memory memmory size * data sample size (s_t, a_t, r_t, s_{t+1})
        self.memory1 = np.zeros((self.MEMORY_CAPACITY1, self.N_STATES * 2 + 2))


        # 自己定义的
        self.memory_label  = torch.zeros((MEMORY_CAPACITY, self.N_STATES))
        self.count_label =  np.zeros((MEMORY_CAPACITY, 1))
        self.memory_state = torch.zeros((MEMORY_CAPACITY, self.N_STATES))


        self.optimizer = torch.optim.Adam([
            {"params": self.eval_net.parameters()},
            # {"params": self.env.W_T_1},
            #             # {"params": self.env.W_T_2},
            #             # {"params": self.env.W_p},
            #             # {"params": self.env.W_p_},
            {"params": self.env.W_u},
            {"params": self.env.model_hyper.parameters()}],
            lr=LR)
            # {"params": self.env.b_T}], lr=LR)

        # 强化学习 高血压优化器
        self.optimizer_hyper = torch.optim.Adam([
            {"params":  self.eval_net.parameters()}
             # {"params": self.env.model_hyper.dense.parameters()}
        ],lr=LR)
        self.loss_func = torch.nn.MSELoss()
        self.loss_func1 = torch.nn.MSELoss()







    def choose_action(self, x):
        action_value=0
        # if np.random.uniform() < self.EPSILON:
        with torch.no_grad():
            action_value = self.eval_net.forward(x)
            action_value = torch.softmax(action_value,dim=1)
            action_value = action_value.numpy()
                #print('action_value', action_value)
            action = np.argmax(action_value)

        # else:
            # action = np.random.randint(0, self.N_ACTIONS)
            # if(action==0):
            #     action_value = [[0.99,0.01]]
            # else:
            #     action_value =  [[0.01,0.99]]

        #print("action ",action)
        #return action,action_value

        return action,action_value

    # def choose_action1(self, x):
    #     '''
    #     \epsilon greedy for generating actions
    #     :param x:
    #     :return: action
    #     '''
    #     action_value=[]
    #     sig_action_value=[]
    #     #print('choose_action',x)
    #     with torch.no_grad():
    #         # 拼接
    #         # x = self.ensemble_state(x[0], x[1])
    #         #print('choose_action', x.shape)
    #         # x shape [1,400]
    #         # action_value = self.eval_net.forward(x)
    #         # action_value = action_value.numpy()
    #         # print("action_value ",action_value)
    #
    #         action_value = self.cls_predict_eval(x)
    #         action_value = action_value.numpy()
    #         print('action_value', action_value)
    #         action = np.argmax(action_value)
    #         #action = np.argmax(action_value.cpu())
    #         action = np.argmax(sig_action_value)
    #
    #     #return action,sig_action_value
    #
    #     return action, sig_action_value

    # 存储状态
    # 存储 进入环境前状态 行动 奖赏 进入环境后状态
    def store_transition(self, s, a, r, s_):
        '''
        ???????
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        '''
        index = self.memory_counter % self.MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = np.hstack((s.detach().numpy(), np.array([[a, r]]), s_.detach().numpy()))
        # self.memory_label[index,:] =
        self.memory_counter += 1

    def store_transition1(self, s, a, r, s_):
        '''
        ???????
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        '''
        index = self.memory_counter1 % self.MEMORY_CAPACITY1
        self.memory1[index, :] = np.hstack((s.detach().numpy(), np.array([[a, r]]), s_.detach().numpy()))
        self.memory_counter1 += 1



    def TD(self, memory):
        b_a = torch.LongTensor(memory[:, self.N_STATES:self.N_STATES + 1])
        b_s = torch.FloatTensor(memory[:, :self.N_STATES])
        b_r = torch.FloatTensor(memory[:, self.N_STATES + 1:self.N_STATES + 2])
        b_s_ = torch.FloatTensor(memory[:, -self.N_STATES:])

        temp  = self.eval_net(b_s)

        #q_eval = self.eval_net(b_s).gather(1, b_a)
        # 选择dim=1那一列
        q_eval = temp.gather(1, b_a)
        #print("temp ", temp," q_eval ",q_eval)
        # 复制target_next 输出值
        # q_next = self.target_net(b_s_).detach()
        q_next = self.target_net(b_s_).detach()
        #print("q_next ",q_next)
        #print("q_next.max(1)[0].view(-1, 1) ",q_next.max(1)[0].view(-1, 1))
        # 选出旧的每一列最大值
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(-1, 1)
        #print("q_eval ",q_eval," q_next ",q_next," q_target ",q_target)

        return q_target, q_eval




    def priority(self, mode='r'):
        '''

        :param b_memory: memory for experience replay
        :param mode: mode='r', select reward-based, mode='TD'; select TD-based
        :return: priority score
        '''

        if mode == 'r':
            b_a = torch.tensor(self.memory[:, self.N_STATES+1:self.N_STATES + 2])
            p_score = b_a
            return p_score.view(-1)
        else:
            q_target, q_eval = self.TD(self.memory)
            p_score = q_target - q_eval
        return p_score.view(-1), q_target, q_eval

    def priority1(self, mode='r'):
        '''

        :param b_memory: memory for experience replay
        :param mode: mode='r', select reward-based, mode='TD'; select TD-based
        :return: priority score
        '''

        if mode == 'r':
            b_a = torch.tensor(self.memory1[:, self.N_STATES+1:self.N_STATES + 2])
            p_score = b_a
            return p_score.view(-1)
        else:
            q_target, q_eval = self.TD(self.memory1)
            p_score = q_target - q_eval
        return p_score.view(-1), q_target, q_eval

    def prob(self, x):
        return F.softmax(x,dim=0)



    def pooling(self, KG):
        '''
        hierarchical pooling for KG state
        :param KG: torch_geometric.data.Data, KG state
        :return: torch.tensor, N*1, one vector for KG
        '''
        entities_cat = KG.x[:KG.num_POI+KG.num_cat]
        entities_loc = torch.cat((KG.x[:KG.num_POI], KG.x[KG.num_POI + KG.num_cat:]), dim=0)

        s_KG_cat = entities_cat.mean(dim=0)
        s_KG_loc = entities_loc.mean(dim=0)

        s_KG = (s_KG_cat + s_KG_loc) / 2

        return s_KG

    def ensemble_state(self, s_u, KG):
        s_KG = self.pooling(KG)
        return torch.cat((s_u.view(1, -1), s_KG.view(1, -1)), dim=1)

    # 学习
    def learn(self,flag=0):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        if self.mode == 'r':

            p_score = self.priority(self.mode)
            p_score1 = self.priority1(self.mode)
            #print('p_score  ', p_score)
            prob = self.prob(p_score)
            prob1 = self.prob(p_score1)
            #print('prob   ',prob)
            # 通过行为 判断随机选取数据
            sample_index = np.random.choice(a=self.MEMORY_CAPACITY, size=self.BATCH_SIZE, p=prob)
            sample_index1 = np.random.choice(a=self.MEMORY_CAPACITY1, size=self.BATCH_SIZE, p=prob1)
            b_memory = self.memory[sample_index, :]
            b_memory1 = self.memory1[sample_index1, :]
            q_target, q_eval = self.TD(b_memory)
            q_target1, q_eval1 = self.TD(b_memory1)
        else:
            p_score, q_target, q_eval = self.priority(self.mode)
            p_score1, q_target1, q_eval1 = self.priority1(self.mode)
            prob = self.prob(p_score)
            prob1 = self.prob(p_score1)
            sample_index = np.random.choice(a=self.MEMORY_CAPACITY, size=self.BATCH_SIZE, p=prob.detach().numpy())
            sample_index1 = np.random.choice(a=self.MEMORY_CAPACITY1, size=self.BATCH_SIZE, p=prob1.detach().numpy())
            q_target = q_target[sample_index]
            q_target1 = q_target1[sample_index1]
            q_eval = q_eval[sample_index]
            q_eval1 = q_eval1[sample_index1]
        #self.cls_predict_target.load_state_dict(self.cls_predict_eval.state_dict())
        #print('q_eval  ',q_eval,'  q_target ',q_target)
        if flag==0:
            self.loss = self.loss_func(q_eval, q_target)
        else :
            self.loss = self.loss_func(q_eval1, q_target1)
        #print("losss:   ",self.loss)
        self.loss.requires_grad_(True)
        #loss1.requires_grad_(True)
        self.optimizer.zero_grad()
        # self.optimizer_hyper.zero_grad()
        self.loss.backward()
        # self.optimizer_hyper.step()
        self.optimizer.step()

        # self.loss = self.loss_func(q_eval1, q_target1)
        # # print("losss:   ",self.loss)
        # self.loss.requires_grad_(True)
        # # loss1.requires_grad_(True)
        # self.optimizer.zero_grad()
        # # self.optimizer_hyper.zero_grad()
        # self.loss.backward()
        # # self.optimizer_hyper.step()
        # self.optimizer.step()
