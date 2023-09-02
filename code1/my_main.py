from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import fitlog
import json
import copy
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import MultipleLocator
import csv
import os
import logging
import argparse
import random
import sys

from tqdm import tqdm, trange
#import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam,SGD
#from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
# from predictive_models import FGM,PGD
from my_predictive_models_new import GBERT_Pretrain_train,FGM,GBERT_Pretrain_train_mod,GBERT_Pretrain_train_mod_1
#import run_pretraining
from run_gbert_dual_age import load_dataset
from my_rlkg_model import RLKGModel
from my_options import Options
import my_utils
from early_stopping import EarlyStopping
from datetime import datetime
from datetime import timedelta


option = Options().parse()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evalutee(model,test_dataloader,device):


    logger.info("***** Running test *****")
    global_step = 0
    model.eval()
    y_preds = []
    y_trues = []
    for test_input in tqdm(test_dataloader, desc="Testing"):
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
        with torch.no_grad():
            bert_predict_tensor,data_len = model(input_ids, age_labels=age_labels, body_tem_labels=body_tem_labels,
                                          pulse_rate_labels=pulse_rate_labels,
                                          breathing_rate_labels=breathing_rate_labels,
                                          lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels, rsbp_labels=rsbp_labels,
                                          rdbp_labels=rdbp_labels,
                                          height_labels=height_labels, weight_labels=weight_labels,
                                          waist_labels=waist_labels, bmi_labels=bmi_labels,
                                          exercise_freq_labels=exercise_freq_labels,
                                          smoking_status_labels=smoking_status_labels,
                                          drinking_freq_labels=drinking_freq_labels,
                                          heart_rate_labels=heart_rate_labels, tcho_labels=tcho_labels,
                                          tg_labels=tg_labels, ldlc_labels=ldlc_labels, hdlc_labels=hdlc_labels,
                                          is_hyper_labels=is_hyper_labels, epoch=global_step,limit_num=9999)
            y_preds.append(t2n(torch.softmax(bert_predict_tensor,dim=1)))
            print('y_preds',y_preds)
            y_trues.append(t2n(is_hyper_labels))


    acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                  0.3)
    my_utils.write_txt('./result.txt', str(acc_container))
    print(acc_container)


def main():
    curr_time = datetime.now()
    today = str((curr_time.strftime("%Y-%m-%d")))


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='Hyper-predict', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data/gonggong',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/gonggong/Hyper-pretraining-gonggong1', type=str, required=False,
                        help="pretraining model")
    parser.add_argument("--train_file", default='hyper-multi-visit.pkl', type=str, required=False,
                        help="training data file.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain",
                        default=True,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.5,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=10,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=100,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()
    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    args.output_dir = os.path.join(args.output_dir, args.model_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer, test_dataset,train_dataset_list,eval_dataset_list = load_dataset(args)
    tokenizer, dataset_list, train_dataset_list, eval_dataset_list = load_dataset(args)


    train_dataloader_list = [DataLoader(train_dataset_list[i],
                                  sampler=RandomSampler(train_dataset_list[i]),
                                  #sampler=SequentialSampler(eval_dataset),
                                  batch_size=1) for i in range(len(train_dataset_list))]
    eval_dataloder_list = [DataLoader(eval_dataset_list[i],
                                 sampler=SequentialSampler(eval_dataset_list[i]),
                                 batch_size=1) for i in range(len(eval_dataset_list))]

    # train_dataloader = DataLoader(train_dataset,
    #                               sampler=RandomSampler(train_dataset),
    #                               #sampler=SequentialSampler(eval_dataset),
    #                               batch_size=1)
    #
    # eval_dataloader = DataLoader(eval_dataset,
    #                              sampler=SequentialSampler(eval_dataset),
    #                              batch_size=1)

    # test_dataloader = DataLoader(test_dataset,
    #                              sampler=SequentialSampler(test_dataset),
    #                              batch_size=1)
    dataloader_list = [DataLoader(dataset_list[i],
                                 sampler=SequentialSampler(dataset_list[i]),
                                 batch_size=1) for i in range(len(dataset_list))]



    # for train_dataloader in train_dataloader_list:
    #     for train_input in tqdm(train_dataloader, desc="Training"):
    #         train_input = tuple(t.to(device) for t in train_input)
    #         input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
    #         lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
    #         waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
    #         drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = train_input
    #         #print('len  is_hyper_labels ',len(is_hyper_labels.squeeze(dim=0)),' is_hyper_labels ',is_hyper_labels.squeeze(dim=0))
    #         if(len(is_hyper_labels.squeeze(dim=0))>10):
    #             print('len  is_hyper_labels ', len(is_hyper_labels.squeeze(dim=0)), ' is_hyper_labels ', is_hyper_labels.squeeze(dim=0))




    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model_env = GBERT_Pretrain_train_mod.from_pretrained(
            args.pretrain_dir, tokenizer=tokenizer, age_voc=tokenizer.age_voc, body_tem_voc=tokenizer.body_tem_voc,
            pulse_rate_voc=tokenizer.pulse_rate_voc, breathing_rate_voc=tokenizer.breathing_rate_voc,
            lsbp_voc=tokenizer.lsbp_voc, ldbp_voc=tokenizer.ldbp_voc, rsbp_voc=tokenizer.rsbp_voc,
            rdbp_voc=tokenizer.rdbp_voc,
            height_voc=tokenizer.height_voc, weight_voc=tokenizer.weight_voc, waist_voc=tokenizer.waist_voc,
            bmi_voc=tokenizer.bmi_voc, exercise_freq_voc=tokenizer.exercise_freq_voc,
            smoking_status_voc=tokenizer.smoking_status_voc, drinking_freq_voc=tokenizer.drinking_freq_voc,
            heart_rate_voc=tokenizer.heart_rate_voc, tcho_voc=tokenizer.tcho_voc, tg_voc=tokenizer.tg_voc,
            ldlc_voc=tokenizer.ldlc_voc, hdlc_voc=tokenizer.hdlc_voc, is_hyper_voc=tokenizer.is_hyper_voc)
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))

        model_env = GBERT_Pretrain_train_mod(config, tokenizer, age_voc=tokenizer.age_voc, body_tem_voc=tokenizer.body_tem_voc,
                                     pulse_rate_voc=tokenizer.pulse_rate_voc,
                                     breathing_rate_voc=tokenizer.breathing_rate_voc, lsbp_voc=tokenizer.lsbp_voc,
                                     ldbp_voc=tokenizer.ldbp_voc, rsbp_voc=tokenizer.rsbp_voc, rdbp_voc=tokenizer.rdbp_voc,
                                     height_voc=tokenizer.height_voc, weight_voc=tokenizer.weight_voc,
                                     waist_voc=tokenizer.waist_voc, bmi_voc=tokenizer.bmi_voc,
                                     exercise_freq_voc=tokenizer.exercise_freq_voc,
                                     smoking_status_voc=tokenizer.smoking_status_voc,
                                     drinking_freq_voc=tokenizer.drinking_freq_voc,
                                     heart_rate_voc=tokenizer.heart_rate_voc, tcho_voc=tokenizer.tcho_voc,
                                     tg_voc=tokenizer.tg_voc, ldlc_voc=tokenizer.ldlc_voc, hdlc_voc=tokenizer.hdlc_voc,
                                     is_hyper_voc=tokenizer.is_hyper_voc)

    model_env.to(device)


    # if args.use_pretrain:
    #     logger.info("Use Pretraining model")
    #     model = GBERT_Pretrain_train.from_pretrained(
    #         args.pretrain_dir, tokenizer=tokenizer,age_voc=tokenizer.age_voc,body_tem_voc=tokenizer.body_tem_voc, pulse_rate_voc=tokenizer.pulse_rate_voc, breathing_rate_voc=tokenizer.breathing_rate_voc, lsbp_voc=tokenizer.lsbp_voc,ldbp_voc=tokenizer.ldbp_voc,rsbp_voc=tokenizer.rsbp_voc,rdbp_voc=tokenizer.rdbp_voc,
    #              height_voc=tokenizer.height_voc, weight_voc=tokenizer.weight_voc,waist_voc=tokenizer.waist_voc,bmi_voc=tokenizer.bmi_voc,exercise_freq_voc=tokenizer.exercise_freq_voc,smoking_status_voc=tokenizer.smoking_status_voc,drinking_freq_voc=tokenizer.drinking_freq_voc,
    #              heart_rate_voc=tokenizer.heart_rate_voc,tcho_voc=tokenizer.tcho_voc,tg_voc=tokenizer.tg_voc,ldlc_voc=tokenizer.ldlc_voc,hdlc_voc=tokenizer.hdlc_voc,is_hyper_voc=tokenizer.is_hyper_voc)
    # else:
    #     config = BertConfig(
    #         vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    #     config.graph = args.graph
    #     model = GBERT_Pretrain_train(config, tokenizer,age_voc=tokenizer.age_voc,body_tem_voc=tokenizer.body_tem_voc, pulse_rate_voc=tokenizer.pulse_rate_voc, breathing_rate_voc=tokenizer.breathing_rate_voc, lsbp_voc=tokenizer.lsbp_voc,ldbp_voc=tokenizer.ldbp_voc,rsbp_voc=tokenizer.rsbp_voc,rdbp_voc=tokenizer.rdbp_voc,
    #              height_voc=tokenizer.height_voc, weight_voc=tokenizer.weight_voc,waist_voc=tokenizer.waist_voc,bmi_voc=tokenizer.bmi_voc,exercise_freq_voc=tokenizer.exercise_freq_voc,smoking_status_voc=tokenizer.smoking_status_voc,drinking_freq_voc=tokenizer.drinking_freq_voc,
    #              heart_rate_voc=tokenizer.heart_rate_voc,tcho_voc=tokenizer.tcho_voc,tg_voc=tokenizer.tg_voc,ldlc_voc=tokenizer.ldlc_voc,hdlc_voc=tokenizer.hdlc_voc,is_hyper_voc=tokenizer.is_hyper_voc)
    #
    #
    # model.to(device)
    #
    # model.train()
    # max_len = 0;
    # for i in range(len(train_dataloader_list)):
    #     for train_input in tqdm(train_dataloader_list[i], desc="Training"):
    #         train_input = tuple(t.to(device) for t in train_input)
    #
    #         input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
    #         lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
    #         waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
    #         drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = train_input
    #
    #         input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
    #         waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
    #             input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
    #             rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
    #             smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
    #                 dim=0)
    #         max_len=max(max_len,is_hyper_labels.size(0))
    #         print(is_hyper_labels.size(0))
    #     for i in range(len(eval_dataloder_list)):
    #         for train_input in tqdm(eval_dataloder_list[i], desc="Training"):
    #             train_input = tuple(t.to(device) for t in train_input)
    #
    #             input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
    #             lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
    #             waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
    #             drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = train_input
    #
    #             input_ids, age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
    #             waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels, tg_labels, ldlc_labels, hdlc_labels, is_hyper_labels = \
    #                 input_ids.squeeze(), age_labels.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
    #                 rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
    #                 smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(), tg_labels.squeeze(), ldlc_labels.squeeze(), hdlc_labels.squeeze(), is_hyper_labels.squeeze(
    #                     dim=0)
    #             max_len = max(max_len, is_hyper_labels.size(0))
    #             print(is_hyper_labels.size(0))
    # print("max_len ",max_len)

    config = BertConfig(
        vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    model_rlkg = RLKGModel(params=option,model_env=model_env,
                           tokenizer=tokenizer, device=device,config=config)
    print("train_dataloader_list LEN ",len(train_dataloader_list))
    print("eval_dataloder_list len ",len(eval_dataloder_list))
    print("data_list len",len(dataset_list))
    logger.info("***** Running train *****")
    cishu=0
    for i in range(len(train_dataloader_list)):
        save_path = "./save_model/"+today+"/"+str(i)
        early_stopping = EarlyStopping(save_path)
        while True:
            model_env.train()
            rx_acc_container = model_rlkg.fit2(train_dataloader=train_dataloader_list[i],device=device)
            rx_save_path = './result/train/'+today+"/"+str(i)
            if not os.path.isdir(rx_save_path):
                os.makedirs(rx_save_path)
            my_utils.write_txt(rx_save_path+ '/train.txt', str(rx_acc_container))
            model_env.eval()
            eval_loss = model_rlkg.evaluate3(eval_dataloader=eval_dataloder_list[i], device=device, temp_count=10000000)
            early_stopping(eval_loss, model_rlkg.dqn.eval_net)
            str_txt = "eval_loss = " + str(eval_loss)
            dir_name = './early_stop/'+today
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            my_utils.write_txt(dir_name+'/early_stopping_eval_bert'+str(i+1)+'.txt', str(eval_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
        # rx_acc_container_test = model_rlkg.test1(test_dataloader=test_dataloader, device=device, temp_count=1000000,save_path=save_path)
        rx_acc_container_test = model_rlkg.test_jiaocha(test_dataloader_list=dataloader_list, device=device, temp_count=1000000,save_path=save_path,i=cishu)
        rx_save_path_test = './result/test/' + today
        if not os.path.isdir(rx_save_path_test):
            os.makedirs(rx_save_path_test)
        my_utils.write_txt(rx_save_path_test + '/test_blanced.txt', str(rx_acc_container_test))
        cishu+=1
    # temp_count=0;
    # temp_train_list = []
    # for j in range(0,1):
    #     for train_input in tqdm(train_dataloader, desc="Training"):
    #         temp_count+=1
    #         train_input = tuple(t.to(device) for t in train_input)
    #         if temp_count%1000!=0:
    #             temp_train_list.append(train_input)
    #
    #         else:
    #             step = 0;
    #             early_stopping = EarlyStopping("../")
    #             while True:
    #                 step+=1;
    #                 # if step>30:break;
    #                 model_env.train()
    #                 model_rlkg.fit4(train_input_list=temp_train_list)
    #                 model_env.eval()
    #                 eval_loss = model_rlkg.evaluate2(eval_dataloader=eval_dataloader, device=device, temp_count=10000)
    #                 early_stopping(eval_loss, model_env)
    #                 str_txt  = "eval_loss = " +str(eval_loss)+"  count = "+ str(temp_count)
    #                 my_utils.write_txt("./early_stop/early_stopping_new1_1000.txt", str_txt)
    #                 if early_stopping.early_stop:
    #                     print("Early stopping")
    #                     break  # 跳出迭代，结束训练
    #             model_rlkg.test(test_dataloader=test_dataloader,device=device,temp_count=10000)
    #             # print("count ======================= "+str(temp_count))
    #             # my_utils.see_group_is_hyper_label(temp_train_list,temp_count)
    #             temp_train_list=[]
    #     step = 0;
    #     early_stopping = EarlyStopping("../")
    #     while True:
    #         step += 1;
    #         # if step>30:break;
    #         model_env.train()
    #         model_rlkg.fit4(train_input_list=temp_train_list)
    #         model_env.eval()
    #         eval_loss = model_rlkg.evaluate2(eval_dataloader=eval_dataloader, device=device, temp_count=10000)
    #         early_stopping(eval_loss, model_env)
    #         str_txt = "eval_loss = " + str(eval_loss) + "  count = " + str(temp_count)
    #         my_utils.write_txt("./early_stop/early_stopping_new1_1000.txt", str_txt)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break  # 跳出迭代，结束训练
    #     model_rlkg.test(test_dataloader=test_dataloader, device=device, temp_count=10000)














































if __name__ == "__main__":
    main()