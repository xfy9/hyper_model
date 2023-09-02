from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import fitlog
import json
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam,SGD
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
# from predictive_models import FGM,PGD
from my_predictive_models_new import GBERT_Pretrain_train,FGM
import run_pretraining

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def text_save(filename,list_data):
    file=open(filename,'a')
    for data in list_data:
        file.write(str(data)+'\n')

def text_save_data(filename,data):
    file=open(filename,'a')
    file.write(str(data)+'\n')

f1_eval_list_run=list()
f1_test_list_run=list()
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

def random_word(token, vocab):
    # for i, _ in enumerate(tokens):
    prob = random.random()
    # mask token with 15% probability
    if prob < 0.15:
        prob /= 0.15

        # 80% randomly change token to mask token
        if prob < 0.8:
            token = "[MASK]"
        # 10% randomly change token to random token
        elif prob < 0.9:
            token = random.choice(list(vocab.word2idx.items()))[0]
        else:
            pass
    else:
        pass

    return token
class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.age_voc = self.add_vocab(os.path.join(data_dir, 'age-vocab.txt'))
        self.body_tem_voc = self.add_vocab(os.path.join(data_dir, 'body-tem-vocab.txt'))
        self.pulse_rate_voc = self.add_vocab(os.path.join(data_dir, 'pulse_rate-vocab.txt'))
        self.breathing_rate_voc = self.add_vocab(os.path.join(data_dir, 'breathing_rate-vocab.txt'))
        self.lsbp_voc = self.add_vocab(os.path.join(data_dir, 'lsbp-vocab.txt'))
        self.ldbp_voc = self.add_vocab(os.path.join(data_dir, 'ldbp-vocab.txt'))
        self.rsbp_voc = self.add_vocab(os.path.join(data_dir, 'rsbp-vocab.txt'))
        self.rdbp_voc = self.add_vocab(os.path.join(data_dir, 'rdbp-vocab.txt'))
        self.height_voc = self.add_vocab(os.path.join(data_dir, 'height-vocab.txt'))
        self.weight_voc = self.add_vocab(os.path.join(data_dir, 'weight-vocab.txt'))
        self.waist_voc = self.add_vocab(os.path.join(data_dir, 'waist-vocab.txt'))
        self.bmi_voc = self.add_vocab(os.path.join(data_dir, 'bmi-vocab.txt'))
        self.exercise_freq_voc = self.add_vocab(os.path.join(data_dir, 'exercise_freq-vocab.txt'))
        self.smoking_status_voc = self.add_vocab(os.path.join(data_dir, 'smoking_status-vocab.txt'))
        self.drinking_freq_voc = self.add_vocab(os.path.join(data_dir, 'drinking_freq-vocab.txt'))
        self.heart_rate_voc = self.add_vocab(os.path.join(data_dir, 'heart_rate-vocab.txt'))
        self.tcho_voc = self.add_vocab(os.path.join(data_dir, 'tcho-vocab.txt'))
        self.tg_voc = self.add_vocab(os.path.join(data_dir, 'tg-vocab.txt'))
        self.ldlc_voc = self.add_vocab(os.path.join(data_dir, 'ldlc-vocab.txt'))
        self.hdlc_voc = self.add_vocab(os.path.join(data_dir, 'hdlc-vocab.txt'))
        self.is_hyper_voc = self.add_vocab(os.path.join(data_dir, 'is_hyper-vocab.txt'))

        # code1 only in multi-visit data

        self.age_voc_multi = Voc()
        self.body_tem_voc_multi = Voc()
        self.pulse_rate_voc_multi = Voc()
        self.breathing_rate_voc_multi = Voc()

        self.lsbp_voc_multi = Voc()
        self.ldbp_voc_multi = Voc()
        self.rsbp_voc_multi = Voc()
        self.rdbp_voc_multi = Voc()
        self.height_voc_multi = Voc()
        self.weight_voc_multi = Voc()
        self.waist_voc_multi = Voc()
        self.bmi_voc_multi = Voc()

        self.exercise_freq_voc_multi = Voc()
        self.smoking_status_voc_multi = Voc()
        self.drinking_freq_voc_multi = Voc()
        self.heart_rate_voc_multi = Voc()
        self.tcho_voc_multi = Voc()
        self.tg_voc_multi = Voc()
        self.ldlc_voc_multi = Voc()
        self.hdlc_voc_multi = Voc()
        self.is_hyper_voc_multi = Voc()

        with open(os.path.join(data_dir, 'age-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.age_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'body-tem-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.body_tem_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'pulse_rate-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.pulse_rate_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'breathing_rate-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.breathing_rate_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'lsbp-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.lsbp_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'ldbp-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.ldbp_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'rsbp-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.rsbp_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'rdbp-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.rdbp_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'height-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.height_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'weight-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.weight_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'waist-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.waist_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'bmi-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.bmi_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'exercise_freq-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.exercise_freq_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'smoking_status-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.smoking_status_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'drinking_freq-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.drinking_freq_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'heart_rate-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.heart_rate_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'tcho-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.tcho_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'tg-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.tg_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'ldlc-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.ldlc_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'hdlc-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.hdlc_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'is_hyper-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.is_hyper_voc_multi.add_sentence([code.rstrip('\n')])

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r', encoding="utf-8") as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
            # print(ids)
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens

def split_str(x):
    return x[0].split(",")

def change_str(x):
    if len(x) > 50:
        return x[:50]
    else:
        return x
# def change_ATC(x):
#     if "110005" in str(x) and len(x)>1:
#         x.remove("110005")
#         return x
#     else:
#         return x
class EHRDataset_pre(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        # data_pd["CHIEF_COMPLAINTS"] = data_pd["CHIEF_COMPLAINTS"].apply(split_str)
        # data_pd["CHIEF_COMPLAINTS"] = data_pd["CHIEF_COMPLAINTS"].apply(change_str)
        # data_pd["ATC4"] = data_pd["ATC4"].apply(change_str)
        # data_pd["ICD9_CODE"] = data_pd["ICD9_CODE"].apply(change_str)
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            admissions = []
            for _, row in data.iterrows():
                admission = list((row['AGE'],row['BODY_TEMPERATURE'],row['PULSE_RATE'],row['BREATHING_RATE'],
                                 row['LSBP'],row['LDBP'],row['RSBP'],row['RDBP'],row['HEIGHT'],row['WEIGHT'],row['WAIST'],row['BMI'],
                                 row['EXERCISE_FREQ_CODE'],row['SMOKING_STATUS_CODE'],row['DRINKING_FREQ_CODE'],row['HEART_RATE'],row['TCHO'],row['TG'],row['LDLC'],row['HDLC'],row['IS_HYPER_3YEARS']))
                admissions.append(admission)
            return admissions

        self.admissions = transform_data(data_pd)

    def __len__(self):
        return len(self.admissions)

    def __getitem__(self, item):
        cur_id = item
        adm = copy.deepcopy(self.admissions[item])

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        """y
        """
        y_age = np.zeros(len(self.tokenizer.age_voc.word2idx))
        y_body_tem = np.zeros(len(self.tokenizer.body_tem_voc.word2idx))
        y_pulse_rate = np.zeros(len(self.tokenizer.pulse_rate_voc.word2idx))
        y_breathing_rate = np.zeros(len(self.tokenizer.breathing_rate_voc.word2idx))
        y_lsbp = np.zeros(len(self.tokenizer.lsbp_voc.word2idx))
        y_ldbp = np.zeros(len(self.tokenizer.ldbp_voc.word2idx))
        y_rsbp = np.zeros(len(self.tokenizer.rsbp_voc.word2idx))
        y_rdbp = np.zeros(len(self.tokenizer.rdbp_voc.word2idx))
        y_height = np.zeros(len(self.tokenizer.height_voc.word2idx))
        y_weight = np.zeros(len(self.tokenizer.weight_voc.word2idx))
        y_waist = np.zeros(len(self.tokenizer.waist_voc.word2idx))
        y_bmi = np.zeros(len(self.tokenizer.bmi_voc.word2idx))
        y_exercise_freq = np.zeros(len(self.tokenizer.exercise_freq_voc.word2idx))
        y_smoking_status = np.zeros(len(self.tokenizer.smoking_status_voc.word2idx))
        y_drinking_freq = np.zeros(len(self.tokenizer.drinking_freq_voc.word2idx))
        y_heart_rate = np.zeros(len(self.tokenizer.heart_rate_voc.word2idx))
        y_tcho = np.zeros(len(self.tokenizer.tcho_voc.word2idx))
        y_tg = np.zeros(len(self.tokenizer.tg_voc.word2idx))
        y_ldlc = np.zeros(len(self.tokenizer.ldlc_voc.word2idx))
        y_hdlc = np.zeros(len(self.tokenizer.hdlc_voc.word2idx))
        y_is_hyper = np.zeros(len(self.tokenizer.is_hyper_voc.word2idx))

        # y_sx = np.zeros(len(self.tokenizer.sx_voc.word2idx))
        # for item in adm[0]:
        # a=adm[0]
        # print(adm[0])
        # b=self.tokenizer.body_tem_voc.word2idx[str(adm[0])]
        # c=y_body_tem[b]


        y_age[self.tokenizer.age_voc.word2idx[str(adm[0])]] = 1
        # print(y_age)
        y_body_tem[self.tokenizer.body_tem_voc.word2idx[str(adm[1])]] = 1
        y_pulse_rate[self.tokenizer.pulse_rate_voc.word2idx[str(adm[2])]] = 1
        y_breathing_rate[self.tokenizer.breathing_rate_voc.word2idx[str(adm[3])]] = 1
        y_lsbp[self.tokenizer.lsbp_voc.word2idx[str(adm[4])]] = 1
        y_ldbp[self.tokenizer.ldbp_voc.word2idx[str(adm[5])]] = 1
        y_rsbp[self.tokenizer.rsbp_voc.word2idx[str(adm[6])]] = 1
        y_rdbp[self.tokenizer.rdbp_voc.word2idx[str(adm[7])]] = 1
        y_height[self.tokenizer.height_voc.word2idx[str(adm[8])]] = 1
        y_weight[self.tokenizer.weight_voc.word2idx[str(adm[9])]] = 1
        y_waist[self.tokenizer.waist_voc.word2idx[str(adm[10])]] = 1
        y_bmi[self.tokenizer.bmi_voc.word2idx[str(adm[11])]] = 1
        y_exercise_freq[self.tokenizer.exercise_freq_voc.word2idx[str(adm[12])]] = 1
        y_smoking_status[self.tokenizer.smoking_status_voc.word2idx[str(adm[13])]] = 1
        y_drinking_freq[self.tokenizer.drinking_freq_voc.word2idx[str(adm[14])]] = 1
        y_heart_rate[self.tokenizer.heart_rate_voc.word2idx[str(adm[15])]] = 1
        y_tcho[self.tokenizer.tcho_voc.word2idx[str(adm[16])]] = 1
        y_tg[self.tokenizer.tg_voc.word2idx[str(adm[17])]] = 1
        y_ldlc[self.tokenizer.ldlc_voc.word2idx[str(adm[18])]] = 1
        y_hdlc[self.tokenizer.hdlc_voc.word2idx[str(adm[19])]] = 1
        y_is_hyper[self.tokenizer.is_hyper_voc.word2idx[str(adm[20])]] = 1
        # for item in adm[2]:
        #     if item == "\n":
        #         item = ""
        #     y_sx[self.tokenizer.sx_voc.word2idx[item]] = 1


        """replace tokens with [MASK]
        """
        adm[0] = random_word(adm[0], self.tokenizer.age_voc)
        adm[1] = random_word(adm[1], self.tokenizer.body_tem_voc)
        adm[2] = random_word(adm[2], self.tokenizer.pulse_rate_voc)
        adm[3] = random_word(adm[3], self.tokenizer.breathing_rate_voc)
        adm[4] = random_word(adm[4], self.tokenizer.lsbp_voc)
        adm[5] = random_word(adm[5], self.tokenizer.ldbp_voc)
        adm[6] = random_word(adm[6], self.tokenizer.rsbp_voc)
        adm[7] = random_word(adm[7], self.tokenizer.rdbp_voc)
        adm[8] = random_word(adm[8], self.tokenizer.height_voc)
        adm[9] = random_word(adm[9], self.tokenizer.weight_voc)
        adm[10] = random_word(adm[10], self.tokenizer.waist_voc)
        adm[11] = random_word(adm[11], self.tokenizer.bmi_voc)
        adm[12] = random_word(adm[12], self.tokenizer.exercise_freq_voc)
        adm[13] = random_word(adm[13], self.tokenizer.smoking_status_voc)
        adm[14] = random_word(adm[14], self.tokenizer.drinking_freq_voc)
        adm[15] = random_word(adm[15], self.tokenizer.heart_rate_voc)
        adm[16] = random_word(adm[16], self.tokenizer.tcho_voc)
        adm[17] = random_word(adm[17], self.tokenizer.tg_voc)
        adm[18] = random_word(adm[18], self.tokenizer.ldlc_voc)
        adm[19] = random_word(adm[19], self.tokenizer.hdlc_voc)
        adm[20] = random_word(adm[20], self.tokenizer.is_hyper_voc)

        """extract input and output tokens
        """
        # random_word
        input_tokens = []  # (3*max_len)
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[0])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[1])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[2])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[3])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[4])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[5])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[6])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[7])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[8])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[9])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[10])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[11])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[12])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[13])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[14])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[15])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[16])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[17])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[18])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[19])], self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max([str(adm[20])], self.seq_len - 1))
        # input_tokens.extend(
        #     ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        # input_tokens.extend(
        #     ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))
        # input_tokens.extend(
        #     ['[CLS]'] + fill_to_max(list(adm[2]), self.seq_len - 1))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        # print("y_age")
        try:
            # print("y_agetry")
            cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.seq_len),
                           torch.tensor(y_age, dtype=torch.float),
                           torch.tensor(y_body_tem, dtype=torch.float),
                           torch.tensor(y_pulse_rate, dtype=torch.float),
                           torch.tensor(y_breathing_rate, dtype=torch.float),
                           torch.tensor(y_lsbp, dtype=torch.float),
                           torch.tensor(y_ldbp, dtype=torch.float),
                           torch.tensor(y_rsbp, dtype=torch.float),
                           torch.tensor(y_rdbp, dtype=torch.float),
                           torch.tensor(y_height, dtype=torch.float),
                           torch.tensor(y_weight, dtype=torch.float),
                           torch.tensor(y_waist, dtype=torch.float),
                           torch.tensor(y_bmi, dtype=torch.float),
                           torch.tensor(y_exercise_freq, dtype=torch.float),
                           torch.tensor(y_smoking_status, dtype=torch.float),
                           torch.tensor(y_drinking_freq, dtype=torch.float),
                           torch.tensor(y_heart_rate, dtype=torch.float),
                           torch.tensor(y_tcho, dtype=torch.float),
                           torch.tensor(y_tg, dtype=torch.float),
                           torch.tensor(y_ldlc, dtype=torch.float),
                           torch.tensor(y_hdlc, dtype=torch.float),
                           torch.tensor(y_is_hyper, dtype=torch.float))
            # print(cur_tensors)
            return cur_tensors
        except:
            print(input_ids)
            print(input_tokens)
            print(len(input_ids),len(input_tokens))



class EHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        # data_pd["CHIEF_COMPLAINTS"] = data_pd["CHIEF_COMPLAINTS"].apply(split_str)
        # data_pd["CHIEF_COMPLAINTS"] = data_pd["CHIEF_COMPLAINTS"].apply(change_str)
        # data_pd["ATC4"] = data_pd["ATC4"].apply(change_str)
        # data_pd["ATC4"] = data_pd["ATC4"].apply(change_ATC)
        # d_ATC=data_pd["ATC4"]
        # i=0
        # for ATC in d_ATC.values:
        #     if "110005" in str(ATC) and len(ATC)>1:
        #         li=ATC.remove("110005")
        #         d_ATC[i] =li
        #         print(ATC)
        #     i=i+1
        # data_pd["ICD9_CODE"] = data_pd["ICD9_CODE"].apply(change_str)
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            records = {}
            for subject_id in data['PERSON_INFO_ID'].unique():
                item_df = data[data['PERSON_INFO_ID'] == subject_id]
                patient = []
                for _, row in item_df.iterrows():
                    admission = list((row['AGE'],row['BODY_TEMPERATURE'],row['PULSE_RATE'],row['BREATHING_RATE'],
                                 row['LSBP'],row['LDBP'],row['RSBP'],row['RDBP'],row['HEIGHT'],row['WEIGHT'],row['WAIST'],row['BMI'],
                                 row['EXERCISE_FREQ_CODE'],row['SMOKING_STATUS_CODE'],row['DRINKING_FREQ_CODE'],row['HEART_RATE'],row['TCHO'],row['TG'],row['LDLC'],row['HDLC'],row['IS_HYPER_3YEARS']))
                    patient.append(admission)
                if len(patient) < 2:
                    continue
                records[subject_id] = patient
            return records

        self.records = transform_data(data_pd)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        subject_id = list(self.records.keys())[item]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (3*max_len*adm)
        output_age_tokens = []  # (adm-1, l)
        output_body_tem_tokens = []  # (adm-1, l)
        output_pulse_rate_tokens = []  # (adm-1, l)
        output_breathing_rate_tokens = []  # (adm-1, l)
        output_lsbp_tokens = []  # (adm-1, l)
        output_ldbp_tokens = []  # (adm-1, l)
        output_rsbp_tokens = []  # (adm-1, l)
        output_rdbp_tokens = []  # (adm-1, l)
        output_height_tokens = []  # (adm-1, l)
        output_weight_tokens = []  # (adm-1, l)
        output_waist_tokens = []  # (adm-1, l)
        output_bmi_tokens = []  # (adm-1, l)
        output_exercise_freq_tokens = []  # (adm-1, l)
        output_smoking_status_tokens = []  # (adm-1, l)
        output_drinking_freq_tokens = []  # (adm-1, l)
        output_heart_rate_tokens = []  # (adm-1, l)
        output_tcho_tokens = []  # (adm-1, l)
        output_tg_tokens = []  # (adm-1, l)
        output_ldlc_tokens = []  # (adm-1, l)
        output_hdlc_tokens = []  # (adm-1, l)
        output_is_hyper_tokens = []  # (adm-1, l)
        for idx, adm in enumerate(self.records[subject_id]):
            # list_0=list(adm[0])
            # fill_to_max(list_0, self.seq_len - 1)
            st=str(adm[0])
            # print(st)
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[0])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[1])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[2])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[3])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[4])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[5])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[6])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[7])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[8])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[9])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[10])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[11])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[12])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[13])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[14])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[15])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[16])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[17])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[18])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[19])], self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max([str(adm[20])], self.seq_len - 1))
            # output_rx_tokens.append(list(adm[1]))

            if idx != 0:
                output_age_tokens.append(str(adm[0]))
                output_body_tem_tokens.append(str(adm[1]))
                output_pulse_rate_tokens.append(str(adm[2]))
                output_breathing_rate_tokens.append(str(adm[3]))
                output_lsbp_tokens.append(str(adm[4]))
                output_ldbp_tokens.append(str(adm[5]))
                output_rsbp_tokens.append(str(adm[6]))
                output_rdbp_tokens.append(str(adm[7]))
                output_height_tokens.append(str(adm[8]))
                output_weight_tokens.append(str(adm[9]))
                output_waist_tokens.append(str(adm[10]))
                output_bmi_tokens.append(str(adm[11]))
                output_exercise_freq_tokens.append(str(adm[12]))
                output_smoking_status_tokens.append(str(adm[13]))
                output_drinking_freq_tokens.append(str(adm[14]))
                output_heart_rate_tokens.append(str(adm[15]))
                output_tcho_tokens.append(str(adm[16]))
                output_tg_tokens.append(str(adm[17]))
                output_ldlc_tokens.append(str(adm[18]))
                output_hdlc_tokens.append(str(adm[19]))
                output_is_hyper_tokens.append(str(adm[20]))
        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # output_dx_labels = []  # (adm-1, dx_voc_size)
        output_age_labels = []  # (adm-1, dx_voc_size)
        output_body_tem_labels = []  # (adm-1, dx_voc_size)
        output_pulse_rate_labels = []  # (adm-1, l)
        output_breathing_rate_labels = []  # (adm-1, l)
        output_lsbp_labels = []  # (adm-1, l)
        output_ldbp_labels = []  # (adm-1, l)
        output_rsbp_labels = []  # (adm-1, l)
        output_rdbp_labels = []  # (adm-1, l)
        output_height_labels = []  # (adm-1, l)
        output_weight_labels = []  # (adm-1, l)
        output_waist_labels = []  # (adm-1, l)
        output_bmi_labels = []  # (adm-1, l)
        output_exercise_freq_labels = []  # (adm-1, l)
        output_smoking_status_labels = []  # (adm-1, l)
        output_drinking_freq_labels = []  # (adm-1, l)
        output_heart_rate_labels = []  # (adm-1, l)
        output_tcho_labels = []  # (adm-1, l)
        output_tg_labels = []  # (adm-1, l)
        output_ldlc_labels = []  # (adm-1, l)
        output_hdlc_labels = []  # (adm-1, l)
        output_is_hyper_labels = []  # (adm-1, l)
        # output_rx_labels = []  # (adm-1, rx_voc_size)
        # output_sx_labels = []  # (adm-1, rx_voc_size)

        age_voc_size = len(self.tokenizer.age_voc_multi.word2idx)
        body_tem_voc_size = len(self.tokenizer.body_tem_voc_multi.word2idx)
        pulse_rate_voc_size = len(self.tokenizer.pulse_rate_voc_multi.word2idx)
        breathing_rate_voc_size = len(self.tokenizer.breathing_rate_voc_multi.word2idx)
        lsbp_voc_size = len(self.tokenizer.lsbp_voc_multi.word2idx)
        ldbp_voc_size = len(self.tokenizer.ldbp_voc_multi.word2idx)
        rsbp_voc_size = len(self.tokenizer.rsbp_voc_multi.word2idx)
        rdbp_voc_size = len(self.tokenizer.rdbp_voc_multi.word2idx)
        height_voc_size = len(self.tokenizer.height_voc_multi.word2idx)
        weight_voc_size = len(self.tokenizer.weight_voc_multi.word2idx)
        waist_voc_size = len(self.tokenizer.waist_voc_multi.word2idx)
        bmi_voc_size = len(self.tokenizer.bmi_voc_multi.word2idx)
        exercise_freq_voc_size = len(self.tokenizer.exercise_freq_voc_multi.word2idx)
        smoking_status_voc_size = len(self.tokenizer.smoking_status_voc_multi.word2idx)
        drinking_freq_voc_size = len(self.tokenizer.drinking_freq_voc_multi.word2idx)
        heart_rate_voc_size = len(self.tokenizer.heart_rate_voc_multi.word2idx)
        tcho_voc_size = len(self.tokenizer.tcho_voc_multi.word2idx)
        tg_voc_size = len(self.tokenizer.tg_voc_multi.word2idx)
        ldlc_voc_size = len(self.tokenizer.ldlc_voc_multi.word2idx)
        hdlc_voc_size = len(self.tokenizer.hdlc_voc_multi.word2idx)
        is_hyper_voc_size = len(self.tokenizer.is_hyper_voc_multi.word2idx)
        for tokens in output_age_tokens:
            tmp_labels = np.zeros(age_voc_size)
            # print(tokens)
            tmp_labels[self.tokenizer.age_voc_multi.word2idx[tokens]]=1
            output_age_labels.append(tmp_labels)

        for tokens in output_body_tem_tokens:
            tmp_labels = np.zeros(body_tem_voc_size)
            # print(tokens)
            tmp_labels[self.tokenizer.body_tem_voc_multi.word2idx[tokens]]=1
            output_body_tem_labels.append(tmp_labels)

        for tokens in output_pulse_rate_tokens:
            tmp_labels = np.zeros(pulse_rate_voc_size)
            tmp_labels[self.tokenizer.pulse_rate_voc_multi.word2idx[tokens]]=1
            output_pulse_rate_labels.append(tmp_labels)
        for tokens in output_breathing_rate_tokens:
            # print(tokens)
            tmp_labels = np.zeros(breathing_rate_voc_size)
            tmp_labels[self.tokenizer.breathing_rate_voc_multi.word2idx[tokens]] = 1
            output_breathing_rate_labels.append(tmp_labels)

        for tokens in output_lsbp_tokens:
            tmp_labels = np.zeros(lsbp_voc_size)
            tmp_labels[self.tokenizer.lsbp_voc_multi.word2idx[tokens]] = 1
            output_lsbp_labels.append(tmp_labels)
        for tokens in output_ldbp_tokens:
            tmp_labels = np.zeros(ldbp_voc_size)
            tmp_labels[self.tokenizer.ldbp_voc_multi.word2idx[tokens]] = 1
            output_ldbp_labels.append(tmp_labels)
        for tokens in output_rsbp_tokens:
            tmp_labels = np.zeros(rsbp_voc_size)
            tmp_labels[self.tokenizer.rsbp_voc_multi.word2idx[tokens]] = 1
            output_rsbp_labels.append(tmp_labels)
        for tokens in output_rdbp_tokens:
            tmp_labels = np.zeros(rdbp_voc_size)
            tmp_labels[self.tokenizer.rdbp_voc_multi.word2idx[tokens]] = 1
            output_rdbp_labels.append(tmp_labels)
        for tokens in output_height_tokens:
            tmp_labels = np.zeros(height_voc_size)
            tmp_labels[self.tokenizer.height_voc_multi.word2idx[tokens]] = 1
            output_height_labels.append(tmp_labels)
        for tokens in output_weight_tokens:
            tmp_labels = np.zeros(weight_voc_size)
            tmp_labels[self.tokenizer.weight_voc_multi.word2idx[tokens]] = 1
            output_weight_labels.append(tmp_labels)
        for tokens in output_waist_tokens:
            tmp_labels = np.zeros(waist_voc_size)
            tmp_labels[self.tokenizer.waist_voc_multi.word2idx[tokens]] = 1
            output_waist_labels.append(tmp_labels)
        for tokens in output_bmi_tokens:
            tmp_labels = np.zeros(bmi_voc_size)
            tmp_labels[self.tokenizer.bmi_voc_multi.word2idx[tokens]] = 1
            output_bmi_labels.append(tmp_labels)
        for tokens in output_exercise_freq_tokens:
            tmp_labels = np.zeros(exercise_freq_voc_size)
            tmp_labels[self.tokenizer.exercise_freq_voc_multi.word2idx[tokens]] = 1
            output_exercise_freq_labels.append(tmp_labels)
        for tokens in output_smoking_status_tokens:
            tmp_labels = np.zeros(smoking_status_voc_size)
            tmp_labels[self.tokenizer.smoking_status_voc_multi.word2idx[tokens]] = 1
            output_smoking_status_labels.append(tmp_labels)
        for tokens in output_drinking_freq_tokens:
            tmp_labels = np.zeros(drinking_freq_voc_size)
            tmp_labels[self.tokenizer.drinking_freq_voc_multi.word2idx[tokens]] = 1
            output_drinking_freq_labels.append(tmp_labels)
        for tokens in output_heart_rate_tokens:
            tmp_labels = np.zeros(heart_rate_voc_size)
            tmp_labels[self.tokenizer.heart_rate_voc_multi.word2idx[tokens]] = 1
            output_heart_rate_labels.append(tmp_labels)

        for tokens in output_tcho_tokens:
            tmp_labels = np.zeros(tcho_voc_size)
            # print(tokens)
            tmp_labels[self.tokenizer.tcho_voc_multi.word2idx[tokens]]=1
            output_tcho_labels.append(tmp_labels)
        for tokens in output_tg_tokens:
            tmp_labels = np.zeros(tg_voc_size)
            # print(tokens)
            tmp_labels[self.tokenizer.tg_voc_multi.word2idx[tokens]]=1
            output_tg_labels.append(tmp_labels)
        for tokens in output_ldlc_tokens:
            tmp_labels = np.zeros(ldlc_voc_size)
            # print(tokens)
            tmp_labels[self.tokenizer.ldlc_voc_multi.word2idx[tokens]]=1
            output_ldlc_labels.append(tmp_labels)
        for tokens in output_hdlc_tokens:
            tmp_labels = np.zeros(hdlc_voc_size)
            # print(tokens)
            tmp_labels[self.tokenizer.hdlc_voc_multi.word2idx[tokens]]=1
            output_hdlc_labels.append(tmp_labels)
        for tokens in output_is_hyper_tokens:
            tmp_labels = np.zeros(is_hyper_voc_size)
            tmp_labels[self.tokenizer.is_hyper_voc_multi.word2idx[tokens]] = 1
            output_is_hyper_labels.append(tmp_labels)
        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("persong_info_id: %s" % subject_id)
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        assert len(input_ids) == (self.seq_len *
                                  21 * len(self.records[subject_id]))
        assert len(output_body_tem_labels) == (len(self.records[subject_id]) - 1)
        # assert len(output_rx_labels) == len(self.records[subject_id])-1

        cur_tensors = (torch.tensor(input_ids).view(-1, self.seq_len),
                       torch.tensor(output_age_labels, dtype=torch.float),
                       torch.tensor(output_body_tem_labels, dtype=torch.float),
                       torch.tensor(output_pulse_rate_labels, dtype=torch.float),
                       torch.tensor(output_breathing_rate_labels, dtype=torch.float),
                       torch.tensor(output_lsbp_labels, dtype=torch.float),
                       torch.tensor(output_ldbp_labels, dtype=torch.float),
                       torch.tensor(output_rsbp_labels, dtype=torch.float),
                       torch.tensor(output_rdbp_labels, dtype=torch.float),
                       torch.tensor(output_height_labels, dtype=torch.float),
                       torch.tensor(output_weight_labels, dtype=torch.float),
                       torch.tensor(output_waist_labels, dtype=torch.float),
                       torch.tensor(output_bmi_labels, dtype=torch.float),
                       torch.tensor(output_exercise_freq_labels, dtype=torch.float),
                       torch.tensor(output_smoking_status_labels, dtype=torch.float),
                       torch.tensor(output_drinking_freq_labels, dtype=torch.float),
                       torch.tensor(output_heart_rate_labels, dtype=torch.float),
                       torch.tensor(output_tcho_labels, dtype=torch.float),
                       torch.tensor(output_tg_labels, dtype=torch.float),
                       torch.tensor(output_ldlc_labels, dtype=torch.float),
                       torch.tensor(output_hdlc_labels, dtype=torch.float),
                       torch.tensor(output_is_hyper_labels, dtype=torch.float))

        return cur_tensors

def load_dataset_pre(args):
    data_dir = args.data_dir
    max_seq_len = args.max_seq_length

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data_multi = pd.read_pickle(os.path.join(
        data_dir, 'hyper-multi-visit.pkl'))
    data_single = pd.read_pickle(
        os.path.join(data_dir, 'hyper-single-visit.pkl'))
    # data_single=data_single.iloc[0:1000]

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'hyper-train-id.txt'),
                os.path.join(data_dir, 'hyper-eval-id.txt'),
                os.path.join(data_dir, 'hyper-test-id2.txt')]

    def load_ids(data, file_name):
        """
        :param data: multi-visit data
        :param file_name:
        :return: raw data form
        """
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(line.rstrip('\n'))
        return data[data['PERSON_INFO_ID'].isin(ids)].reset_index(drop=True)
    # return tokenizer, \
    #     EHRDataset(load_ids(data_multi, ids_file[1]), tokenizer, max_seq_len), \
    #     EHRDataset(load_ids(data_multi, ids_file[1]), tokenizer, max_seq_len), \
    #     EHRDataset(load_ids(data_multi, ids_file[2]), tokenizer, max_seq_len)
    return tokenizer, \
        EHRDataset_pre(pd.concat([data_single, load_ids(
            data_multi, ids_file[0])]), tokenizer, max_seq_len), \
        EHRDataset_pre(load_ids(data_multi, ids_file[1]), tokenizer, max_seq_len), \
        EHRDataset_pre(load_ids(data_multi, ids_file[2]), tokenizer, max_seq_len)
def load_dataset(args):
    data_dir = args.data_dir
    max_seq_len = args.max_seq_length

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data = pd.read_pickle(os.path.join(data_dir, 'hyper-multi-visit.pkl'))
    # data_single = pd.read_pickle(os.path.join(data_dir, 'hyper-single-visit.pkl'))

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'hyper-train-id.txt'),
                os.path.join(data_dir, 'hyper-eval-id.txt'),
                os.path.join(data_dir, 'hyper-test-id2.txt')]

    def load_ids(data, file_name):
        """
        :param data: multi-visit data
        :param file_name:
        :return: raw data form
        """
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(line.rstrip('\n'))
        tmp = data[data['PERSON_INFO_ID'].isin(ids)].reset_index(drop=True)
        return tmp

    # tokenizer, tuple(map(lambda x: EHRDataset(load_ids(data, x), tokenizer, max_seq_len), ids_file))
    # return tokenizer, tuple(map(lambda x: EHRDataset(load_ids(data, x), tokenizer, max_seq_len), ids_file))
    return tokenizer, \
           (EHRDataset(load_ids(data, ids_file[0]), tokenizer, max_seq_len), \
           EHRDataset(load_ids(data, ids_file[1]), tokenizer, max_seq_len), \
           EHRDataset(load_ids(data, ids_file[2]), tokenizer, max_seq_len))

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='Hyper-predict', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data/hyper',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/Hyper-predict', type=str, required=False,
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
                        default=False,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.3,
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
                        default=100,
                        type=int,
                        help="Total batch size for training.")
    args_pre=parser.parse_args()
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

    print("Loading Dataset")
    print("1")
    tokenizer, train_dataset_pre, eval_dataset, test_dataset = load_dataset_pre(args_pre)
    print("2")
    tokenizer, (train_dataset, eval_dataset, test_dataset) = load_dataset(args)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=1)
    train_dataloader_pre = DataLoader(train_dataset_pre,
                                  sampler=RandomSampler(train_dataset_pre),
                                  batch_size=20)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=SequentialSampler(eval_dataset),
                                 batch_size=1)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=1)

    print('Loading Model: ' + args.model_name)
    # config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx), side_len=train_dataset.side_len)
    # config.graph = args.graph
    # model = SeperateBertTransModel(config, tokenizer.dx_voc, tokenizer.rx_voc)
    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model = GBERT_Pretrain_train.from_pretrained(
            args.pretrain_dir, tokenizer=tokenizer,age_voc=tokenizer.age_voc,body_tem_voc=tokenizer.body_tem_voc, pulse_rate_voc=tokenizer.pulse_rate_voc, breathing_rate_voc=tokenizer.breathing_rate_voc, lsbp_voc=tokenizer.lsbp_voc,ldbp_voc=tokenizer.ldbp_voc,rsbp_voc=tokenizer.rsbp_voc,rdbp_voc=tokenizer.rdbp_voc,
                 height_voc=tokenizer.height_voc, weight_voc=tokenizer.weight_voc,waist_voc=tokenizer.waist_voc,bmi_voc=tokenizer.bmi_voc,exercise_freq_voc=tokenizer.exercise_freq_voc,smoking_status_voc=tokenizer.smoking_status_voc,drinking_freq_voc=tokenizer.drinking_freq_voc,
                 heart_rate_voc=tokenizer.heart_rate_voc,tcho_voc=tokenizer.tcho_voc,tg_voc=tokenizer.tg_voc,ldlc_voc=tokenizer.ldlc_voc,hdlc_voc=tokenizer.hdlc_voc,is_hyper_voc=tokenizer.is_hyper_voc)
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
        config.graph = args.graph
        model = GBERT_Pretrain_train(config, tokenizer,age_voc=tokenizer.age_voc,body_tem_voc=tokenizer.body_tem_voc, pulse_rate_voc=tokenizer.pulse_rate_voc, breathing_rate_voc=tokenizer.breathing_rate_voc, lsbp_voc=tokenizer.lsbp_voc,ldbp_voc=tokenizer.ldbp_voc,rsbp_voc=tokenizer.rsbp_voc,rdbp_voc=tokenizer.rdbp_voc,
                 height_voc=tokenizer.height_voc, weight_voc=tokenizer.weight_voc,waist_voc=tokenizer.waist_voc,bmi_voc=tokenizer.bmi_voc,exercise_freq_voc=tokenizer.exercise_freq_voc,smoking_status_voc=tokenizer.smoking_status_voc,drinking_freq_voc=tokenizer.drinking_freq_voc,
                 heart_rate_voc=tokenizer.heart_rate_voc,tcho_voc=tokenizer.tcho_voc,tg_voc=tokenizer.tg_voc,ldlc_voc=tokenizer.ldlc_voc,hdlc_voc=tokenizer.hdlc_voc,is_hyper_voc=tokenizer.is_hyper_voc)


    model.to(device)

    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    rx_output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin")

    # Prepare optimizer
    # num_train_optimization_steps = int(
    #     len(train_dataset) / args.train_batch_size) * args.num_train_epochs
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_optimization_steps)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    optimizer2 = SGD(model.parameters(), lr=5e-8)
    optimizer3 = SGD(model.parameters(), lr=3.5e-4)

    global_step = 0
    if args.do_train:
        writer = SummaryWriter(args.output_dir)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", 1)

        dx_acc_best, rx_acc_best = 0, 0
        acc_name = 'prauc'
        dx_history = {'prauc': []}
        rx_history = {'prauc': []}

        # fgm = FGM(model)
        loss_list=list()
        loss_list_adv = list()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            tr_loss_adv=0
            nb_tr_examples, nb_tr_steps = 0, 0
            # #进度条-训练
            # prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            #进度条-预训练
            prog_iter_pre=tqdm(train_dataloader_pre, leave=False, desc='Training_pre')

            model.train()
            for _, batch in enumerate(prog_iter_pre):
                batch_pre=batch
                # batch_train = batch[1]

                batch_pre = tuple(t.to(device) for t in batch_pre)
                # batch_train = tuple(t.to(device) for t in batch_train)

                # input_ids, body_tem_labels, pulse_rate_labels,breathing_rate_labels,\
                # lsbp_labels, ldbp_labels,rsbp_labels,rdbp_labels, height_labels,weight_labels,\
                # waist_labels, bmi_labels,exercise_freq_labels, smoking_status_labels,\
                # drinking_freq_labels,heart_rate_labels,is_hyper_labels= batch_train

                input_ids_pre,age_labels_pre, body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre, lsbp_labels_pre, ldbp_labels_pre, \
                rsbp_labels_pre, rdbp_labels_pre, height_labels_pre, weight_labels_pre, waist_labels_pre, bmi_labels_pre, exercise_freq_labels_pre, \
                smoking_status_labels_pre, drinking_freq_labels_pre, heart_rate_labels_pre,tcho_labels_pre,tg_labels_pre,ldlc_labels_pre,hdlc_labels_pre, is_hyper_labels_pre = batch_pre



                # input_ids, body_tem_labels, pulse_rate_labels,breathing_rate_labels,\
                # lsbp_labels, ldbp_labels,rsbp_labels,rdbp_labels, height_labels,weight_labels,\
                # waist_labels, bmi_labels,exercise_freq_labels, smoking_status_labels,\
                # drinking_freq_labels,heart_rate_labels,is_hyper_labels = input_ids.squeeze(), body_tem_labels.squeeze(), pulse_rate_labels.squeeze(dim=0),breathing_rate_labels.squeeze(dim=0),\
                # lsbp_labels.squeeze(dim=0), ldbp_labels.squeeze(dim=0),rsbp_labels.squeeze(dim=0),rdbp_labels.squeeze(dim=0),\
                # height_labels.squeeze(dim=0),weight_labels.squeeze(dim=0),waist_labels.squeeze(dim=0), bmi_labels.squeeze(dim=0),\
                # exercise_freq_labels.squeeze(dim=0), smoking_status_labels.squeeze(dim=0),\
                # drinking_freq_labels.squeeze(dim=0),heart_rate_labels.squeeze(dim=0),is_hyper_labels.squeeze(dim=0)

                input_ids_pre,age_labels_pre,body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre, \
                lsbp_labels_pre, ldbp_labels_pre, rsbp_labels_pre, rdbp_labels_pre, height_labels_pre, weight_labels_pre, \
                waist_labels_pre, bmi_labels_pre, exercise_freq_labels_pre, smoking_status_labels_pre, \
                drinking_freq_labels_pre, heart_rate_labels_pre,tcho_labels_pre,tg_labels_pre,ldlc_labels_pre,hdlc_labels_pre, is_hyper_labels_pre = input_ids_pre.squeeze(),age_labels_pre.squeeze(),body_tem_labels_pre.squeeze(), pulse_rate_labels_pre.squeeze(
                    dim=0), breathing_rate_labels_pre.squeeze(dim=0), \
                                                                           lsbp_labels_pre.squeeze(
                                                                               dim=0), ldbp_labels_pre.squeeze(
                    dim=0), rsbp_labels_pre.squeeze(dim=0), rdbp_labels_pre.squeeze(dim=0), \
                                                                           height_labels_pre.squeeze(
                                                                               dim=0), weight_labels_pre.squeeze(
                    dim=0), waist_labels_pre.squeeze(dim=0), bmi_labels_pre.squeeze(dim=0), \
                                                                           exercise_freq_labels_pre.squeeze(
                                                                               dim=0), smoking_status_labels_pre.squeeze(
                    dim=0), \
                                                                           drinking_freq_labels_pre.squeeze(
                                                                               dim=0), heart_rate_labels_pre.squeeze(
                    dim=0), tcho_labels_pre.squeeze(dim=0),tg_labels_pre.squeeze(dim=0),ldlc_labels_pre.squeeze(dim=0),hdlc_labels_pre.squeeze(dim=0),is_hyper_labels_pre.squeeze(dim=0)

                # loss, rx_logits = model(input_ids, body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                #                         lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                #                         height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                #                         exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                #                         heart_rate_labels=heart_rate_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)
                # loss.backward(retain_graph=True)
                # fgm.attack()  # 在embedding上添加对抗扰动

                # loss_adv, rx_logits = model(input_ids, body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                #                         lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                #                         height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                #                         exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                #                         heart_rate_labels=heart_rate_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)
                # print(age_labels_pre)
                loss_pre, age2age,body_tem2body_tem, pulse_rate2pulse_rate, breathing_rate2breathing_rate, lsbp2lsbp, ldbp2ldbp, rsbp2rsbp, rdbp2rdbp, height2height, weight2weight, \
                waist2waist, bmi2bmi, exercise_freq2exercise_freq, smoking_status2smoking_status, drinking_freq2drinking_freq, heart_rate2heart_rate, tcho2tcho,tg2tg,ldlc2ldlc,hdlc2hdlc,is_hyper2is_hyper, \
                age2is_hyper,body_tem2is_hyper, pulse_rate2is_hyper, breathing_rate2is_hyper, lsbp2is_hyper, ldbp2is_hyper, rsbp2is_hyper, rdbp2is_hyper, height2is_hyper, weight2is_hyper, \
                waist2is_hyper, bmi2is_hyper, exercise_freq2is_hyper, smoking_status2is_hyper, drinking_freq2is_hyper, heart_rate2is_hyper,tcho2is_hyper,tg2is_hyper,ldlc2is_hyper,hdlc2is_hyper = model(
                    input_ids_pre, age_labels_pre,body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre,
                    lsbp_labels_pre, ldbp_labels_pre, rsbp_labels_pre, rdbp_labels_pre,
                    height_labels_pre, weight_labels_pre, waist_labels_pre, bmi_labels_pre,
                    exercise_freq_labels_pre, smoking_status_labels_pre, drinking_freq_labels_pre,
                    heart_rate_labels_pre,tcho_labels_pre,tg_labels_pre,ldlc_labels_pre,hdlc_labels_pre,is_hyper_labels_pre,global_step,flag=0)

                # loss_total=0.1*loss_pre+2*loss
                loss_pre.backward()
                # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # fgm.restore()  # 恢复embedding参数

                tr_loss += loss_pre.item()
                # tr_loss_adv += loss_adv.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter_pre.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))
                loss_list.append('%.4f' % (tr_loss / nb_tr_steps))
                # loss_list_adv.append('%.4f' % (tr_loss_adv / nb_tr_steps))

                optimizer2.step()
                optimizer2.zero_grad()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            tr_loss_adv=0
            nb_tr_examples, nb_tr_steps = 0, 0
            # #进度条-训练
            prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            #进度条-预训练
            # prog_iter_pre=tqdm(train_dataloader_pre, leave=False, desc='Training_pre')

            model.train()
            for _, batch in enumerate(prog_iter):
                batch_train=batch
                # batch_train = batch[1]

                # batch_pre = tuple(t.to(device) for t in batch_pre)
                batch_train = tuple(t.to(device) for t in batch_train)

                input_ids, age_labels,body_tem_labels, pulse_rate_labels,breathing_rate_labels,\
                lsbp_labels, ldbp_labels,rsbp_labels,rdbp_labels, height_labels,weight_labels,\
                waist_labels, bmi_labels,exercise_freq_labels, smoking_status_labels,\
                drinking_freq_labels,heart_rate_labels,tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels= batch_train

                # input_ids_pre, body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre, lsbp_labels_pre, ldbp_labels_pre, \
                # rsbp_labels_pre, rdbp_labels_pre, height_labels_pre, weight_labels_pre, waist_labels_pre, bmi_labels_pre, exercise_freq_labels_pre, \
                # smoking_status_labels_pre, drinking_freq_labels_pre, heart_rate_labels_pre, is_hyper_labels_pre = batch_pre


                input_ids, age_labels,body_tem_labels, pulse_rate_labels,breathing_rate_labels,\
                lsbp_labels, ldbp_labels,rsbp_labels,rdbp_labels, height_labels,weight_labels,\
                waist_labels, bmi_labels,exercise_freq_labels, smoking_status_labels,\
                drinking_freq_labels,heart_rate_labels,tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels = input_ids.squeeze(), age_labels.squeeze(dim=0),body_tem_labels.squeeze(), pulse_rate_labels.squeeze(dim=0),breathing_rate_labels.squeeze(dim=0),\
                lsbp_labels.squeeze(dim=0), ldbp_labels.squeeze(dim=0),rsbp_labels.squeeze(dim=0),rdbp_labels.squeeze(dim=0),\
                height_labels.squeeze(dim=0),weight_labels.squeeze(dim=0),waist_labels.squeeze(dim=0), bmi_labels.squeeze(dim=0),\
                exercise_freq_labels.squeeze(dim=0), smoking_status_labels.squeeze(dim=0),\
                drinking_freq_labels.squeeze(dim=0),heart_rate_labels.squeeze(dim=0),tcho_labels.squeeze(dim=0),tg_labels.squeeze(dim=0),ldlc_labels.squeeze(dim=0),hdlc_labels.squeeze(dim=0),is_hyper_labels.squeeze(dim=0)

                # input_ids_pre,body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre, \
                # lsbp_labels_pre, ldbp_labels_pre, rsbp_labels_pre, rdbp_labels_pre, height_labels_pre, weight_labels_pre, \
                # waist_labels_pre, bmi_labels_pre, exercise_freq_labels_pre, smoking_status_labels_pre, \
                # drinking_freq_labels_pre, heart_rate_labels_pre, is_hyper_labels_pre = input_ids_pre.squeeze(),body_tem_labels_pre.squeeze(), pulse_rate_labels_pre.squeeze(
                #     dim=0), breathing_rate_labels_pre.squeeze(dim=0), \
                #                                                            lsbp_labels_pre.squeeze(
                #                                                                dim=0), ldbp_labels_pre.squeeze(
                #     dim=0), rsbp_labels_pre.squeeze(dim=0), rdbp_labels_pre.squeeze(dim=0), \
                #                                                            height_labels_pre.squeeze(
                #                                                                dim=0), weight_labels_pre.squeeze(
                #     dim=0), waist_labels_pre.squeeze(dim=0), bmi_labels_pre.squeeze(dim=0), \
                #                                                            exercise_freq_labels_pre.squeeze(
                #                                                                dim=0), smoking_status_labels_pre.squeeze(
                #     dim=0), \
                #                                                            drinking_freq_labels_pre.squeeze(
                #                                                                dim=0), heart_rate_labels_pre.squeeze(
                #     dim=0), is_hyper_labels_pre.squeeze(dim=0)

                loss, rx_logits = model(input_ids, age_labels=age_labels,body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                                        lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                                        height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                                        exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                                        heart_rate_labels=heart_rate_labels,tcho_labels=tcho_labels,tg_labels=tg_labels,ldlc_labels=ldlc_labels,hdlc_labels=hdlc_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)
                # loss.backward(retain_graph=True)
                # fgm.attack()  # 在embedding上添加对抗扰动

                # loss_adv, rx_logits = model(input_ids, body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                #                         lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                #                         height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                #                         exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                #                         heart_rate_labels=heart_rate_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)

                # loss_pre, body_tem2body_tem, pulse_rate2pulse_rate, breathing_rate2breathing_rate, lsbp2lsbp, ldbp2ldbp, rsbp2rsbp, rdbp2rdbp, height2height, weight2weight, \
                # waist2waist, bmi2bmi, exercise_freq2exercise_freq, smoking_status2smoking_status, drinking_freq2drinking_freq, heart_rate2heart_rate, is_hyper2is_hyper, \
                # body_tem2is_hyper, pulse_rate2is_hyper, breathing_rate2is_hyper, lsbp2is_hyper, ldbp2is_hyper, rsbp2is_hyper, rdbp2is_hyper, height2is_hyper, weight2is_hyper, \
                # waist2is_hyper, bmi2is_hyper, exercise_freq2is_hyper, smoking_status2is_hyper, drinking_freq2is_hyper, heart_rate2is_hyper = model(
                #     input_ids_pre, body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre,
                #     lsbp_labels_pre, ldbp_labels_pre, rsbp_labels_pre, rdbp_labels_pre,
                #     height_labels_pre, weight_labels_pre, waist_labels_pre, bmi_labels_pre,
                #     exercise_freq_labels_pre, smoking_status_labels_pre, drinking_freq_labels_pre,
                #     heart_rate_labels_pre, is_hyper_labels_pre,global_step,flag=0)

                # loss_total=0.1*loss_pre+2*loss
                loss.backward()
                # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # fgm.restore()  # 恢复embedding参数

                tr_loss += loss.item()
                # tr_loss_adv += loss_adv.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))
                loss_list.append('%.4f' % (tr_loss / nb_tr_steps))
                # loss_list_adv.append('%.4f' % (tr_loss_adv / nb_tr_steps))

                optimizer3.step()
                optimizer3.zero_grad()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            tr_loss_adv=0
            nb_tr_examples, nb_tr_steps = 0, 0
            #进度条-训练
            prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            #进度条-预训练
            prog_iter_pre=tqdm(train_dataloader_pre, leave=False, desc='Training_pre')

            model.train()
            for _, batch in enumerate(zip(prog_iter_pre,prog_iter)):
                batch_pre=batch[0]
                batch_train = batch[1]

                batch_pre = tuple(t.to(device) for t in batch_pre)
                batch_train = tuple(t.to(device) for t in batch_train)

                input_ids,age_labels, body_tem_labels, pulse_rate_labels,breathing_rate_labels,\
                lsbp_labels, ldbp_labels,rsbp_labels,rdbp_labels, height_labels,weight_labels,\
                waist_labels, bmi_labels,exercise_freq_labels, smoking_status_labels,\
                drinking_freq_labels,heart_rate_labels,tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels= batch_train

                input_ids_pre, age_labels_pre,body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre, lsbp_labels_pre, ldbp_labels_pre, \
                rsbp_labels_pre, rdbp_labels_pre, height_labels_pre, weight_labels_pre, waist_labels_pre, bmi_labels_pre, exercise_freq_labels_pre, \
                smoking_status_labels_pre, drinking_freq_labels_pre, heart_rate_labels_pre, tcho_labels_pre,tg_labels_pre,ldlc_labels_pre,hdlc_labels_pre,is_hyper_labels_pre = batch_pre


                input_ids, age_labels,body_tem_labels, pulse_rate_labels,breathing_rate_labels,\
                lsbp_labels, ldbp_labels,rsbp_labels,rdbp_labels, height_labels,weight_labels,\
                waist_labels, bmi_labels,exercise_freq_labels, smoking_status_labels,\
                drinking_freq_labels,heart_rate_labels,tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels = input_ids.squeeze(), age_labels.squeeze(),body_tem_labels.squeeze(), pulse_rate_labels.squeeze(dim=0),breathing_rate_labels.squeeze(dim=0),\
                lsbp_labels.squeeze(dim=0), ldbp_labels.squeeze(dim=0),rsbp_labels.squeeze(dim=0),rdbp_labels.squeeze(dim=0),\
                height_labels.squeeze(dim=0),weight_labels.squeeze(dim=0),waist_labels.squeeze(dim=0), bmi_labels.squeeze(dim=0),\
                exercise_freq_labels.squeeze(dim=0), smoking_status_labels.squeeze(dim=0),\
                drinking_freq_labels.squeeze(dim=0),heart_rate_labels.squeeze(dim=0),tcho_labels.squeeze(dim=0),tg_labels.squeeze(dim=0),ldlc_labels.squeeze(dim=0),hdlc_labels.squeeze(dim=0),is_hyper_labels.squeeze(dim=0)

                input_ids_pre,age_labels_pre,body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre, \
                lsbp_labels_pre, ldbp_labels_pre, rsbp_labels_pre, rdbp_labels_pre, height_labels_pre, weight_labels_pre, \
                waist_labels_pre, bmi_labels_pre, exercise_freq_labels_pre, smoking_status_labels_pre, \
                drinking_freq_labels_pre, heart_rate_labels_pre, tcho_labels_pre,tg_labels_pre,ldlc_labels_pre,hdlc_labels_pre,is_hyper_labels_pre = input_ids_pre.squeeze(),age_labels_pre.squeeze(),body_tem_labels_pre.squeeze(), pulse_rate_labels_pre.squeeze(
                    dim=0), breathing_rate_labels_pre.squeeze(dim=0), \
                                                                           lsbp_labels_pre.squeeze(
                                                                               dim=0), ldbp_labels_pre.squeeze(
                    dim=0), rsbp_labels_pre.squeeze(dim=0), rdbp_labels_pre.squeeze(dim=0), \
                                                                           height_labels_pre.squeeze(
                                                                               dim=0), weight_labels_pre.squeeze(
                    dim=0), waist_labels_pre.squeeze(dim=0), bmi_labels_pre.squeeze(dim=0), \
                                                                           exercise_freq_labels_pre.squeeze(
                                                                               dim=0), smoking_status_labels_pre.squeeze(
                    dim=0), \
                                                                           drinking_freq_labels_pre.squeeze(
                                                                               dim=0), heart_rate_labels_pre.squeeze(
                    dim=0), tcho_labels_pre.squeeze(dim=0),tg_labels_pre.squeeze(dim=0),ldlc_labels_pre.squeeze(dim=0),hdlc_labels_pre.squeeze(dim=0),is_hyper_labels_pre.squeeze(dim=0)

                loss, rx_logits = model(input_ids, age_labels=age_labels,body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                                        lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                                        height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                                        exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                                        heart_rate_labels=heart_rate_labels,tcho_labels=tcho_labels,tg_labels=tg_labels,ldlc_labels=ldlc_labels,hdlc_labels=hdlc_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)
                # loss.backward(retain_graph=True)
                # fgm.attack()  # 在embedding上添加对抗扰动

                # loss_adv, rx_logits = model(input_ids, body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                #                         lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                #                         height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                #                         exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                #                         heart_rate_labels=heart_rate_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)

                loss_pre, age2age,body_tem2body_tem, pulse_rate2pulse_rate, breathing_rate2breathing_rate, lsbp2lsbp, ldbp2ldbp, rsbp2rsbp, rdbp2rdbp, height2height, weight2weight, \
                waist2waist, bmi2bmi, exercise_freq2exercise_freq, smoking_status2smoking_status, drinking_freq2drinking_freq, heart_rate2heart_rate, tcho2tcho,tg2tg,ldlc2ldlc,hdlc2hdlc,is_hyper2is_hyper, \
                age2is_hyper,body_tem2is_hyper, pulse_rate2is_hyper, breathing_rate2is_hyper, lsbp2is_hyper, ldbp2is_hyper, rsbp2is_hyper, rdbp2is_hyper, height2is_hyper, weight2is_hyper, \
                waist2is_hyper, bmi2is_hyper, exercise_freq2is_hyper, smoking_status2is_hyper, drinking_freq2is_hyper, heart_rate2is_hyper,tcho2is_hyper,tg2is_hyper,ldlc2is_hyper,hdlc2is_hyper = model(
                    input_ids_pre, age_labels_pre,body_tem_labels_pre, pulse_rate_labels_pre, breathing_rate_labels_pre,
                    lsbp_labels_pre, ldbp_labels_pre, rsbp_labels_pre, rdbp_labels_pre,
                    height_labels_pre, weight_labels_pre, waist_labels_pre, bmi_labels_pre,
                    exercise_freq_labels_pre, smoking_status_labels_pre, drinking_freq_labels_pre,
                    heart_rate_labels_pre, tcho_labels_pre,tg_labels_pre,ldlc_labels_pre,hdlc_labels_pre,is_hyper_labels_pre,global_step,flag=0)

                loss_total=0.1*loss_pre+loss
                loss_total.backward()
                # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # fgm.restore()  # 恢复embedding参数

                tr_loss += loss_total.item()
                # tr_loss_adv += loss_adv.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))
                loss_list.append('%.4f' % (tr_loss / nb_tr_steps))
                # loss_list_adv.append('%.4f' % (tr_loss_adv / nb_tr_steps))

                optimizer.step()
                optimizer.zero_grad()


            writer.add_scalar('train/loss', tr_loss / nb_tr_steps, global_step)
            global_step += 1

            # text_save_data('wuduikang/txt_loss_wuduikang_epoch_nograph.txt',('%.4f' % (tr_loss / nb_tr_steps)))
            text_save('wuduikang/dual_age_train_loss_epoch=4_1.txt', loss_list)

            if args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                model.eval()
                dx_y_preds = []
                dx_y_trues = []
                rx_y_preds = []
                rx_y_trues = []
                for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
                    eval_input = tuple(t.to(device) for t in eval_input)

                    input_ids,age_labels, body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
                    lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
                    waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
                    drinking_freq_labels, heart_rate_labels, tcho_labels,tg_labels,ldlc_labels,hdlc_labesl,is_hyper_labels = eval_input

                    input_ids, age_labels,body_tem_labels, pulse_rate_labels, breathing_rate_labels,lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
                    waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels,drinking_freq_labels, heart_rate_labels, tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels = \
                        input_ids.squeeze(), age_labels.squeeze(),body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(),lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                        rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(),exercise_freq_labels.squeeze(),\
                        smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(),tg_labels.squeeze(),ldlc_labels.squeeze(),hdlc_labels.squeeze(),is_hyper_labels.squeeze(dim=0)

                    # input_ids, dx_labels, rx_labels,sx_labels = eval_input
                    # input_ids, dx_labels, rx_labels,sx_labels = input_ids.squeeze(
                    # ), dx_labels.squeeze(), rx_labels.squeeze(dim=0),sx_labels.squeeze(dim=0)
                    with torch.no_grad():
                        loss, is_hyper_logits = model(input_ids, age_labels=age_labels,body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                                        lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                                        height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                                        exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                                        heart_rate_labels=heart_rate_labels,tcho_labels=tcho_labels,tg_labels=tg_labels,ldlc_labels=ldlc_labels,hdlc_labels=hdlc_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)
                        rx_y_preds.append(t2n(torch.sigmoid(is_hyper_logits)))
                        rx_y_trues.append(t2n(is_hyper_labels))
                        # dx_y_preds.append(t2n(torch.sigmoid(dx_logits)))
                        # dx_y_trues.append(
                        #     t2n(dx_labels.view(-1, len(tokenizer.dx_voc.word2idx))))
                        # rx_y_preds.append(t2n(torch.sigmoid(rx_logits))[
                        #                   :, tokenizer.rx_singe2multi])
                        # rx_y_trues.append(
                        #     t2n(rx_labels)[:, tokenizer.rx_singe2multi])

                print('')
                # dx_acc_container = metric_report(np.concatenate(dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0),
                #                                  args.therhold)
                rx_acc_container = metric_report(np.concatenate(rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0),
                                                 args.therhold)
                # jsObj = json.dumps(rx_acc_container)
                # fileObject = open('jsonFile_eval_1.json', 'a')
                # fileObject.write(jsObj)
                # fileObject.close()
                # f1_eval_list_run.append(rx_acc_container['f1'])
                for k, v in rx_acc_container.items():
                    writer.add_scalar(
                        'eval/{}'.format(k), v, global_step)

                if rx_acc_container[acc_name] > rx_acc_best:
                    rx_acc_best = rx_acc_container[acc_name]
                    # save model
                    torch.save(model_to_save.state_dict(),
                               rx_output_model_file)

        with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
            fout.write(model.config.to_json_string())

    if args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", 1)

        def test(task=0):
            # Load a trained model that you have fine-tuned
            model_state_dict = torch.load(rx_output_model_file)
            model.load_state_dict(model_state_dict)
            model.to(device)

            model.eval()
            y_preds = []
            y_trues = []
            test_loss_1 = 0
            # tr_loss_adv = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            test_loss_list = list()
            for test_input in tqdm(test_dataloader, desc="Testing"):
                test_input = tuple(t.to(device) for t in test_input)
                # input_ids, dx_labels, rx_labels, sx_labels = test_input
                input_ids, age_labels,body_tem_labels, pulse_rate_labels, breathing_rate_labels, \
                lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
                waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, \
                drinking_freq_labels, heart_rate_labels, tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels = test_input

                input_ids, age_labels,body_tem_labels, pulse_rate_labels, breathing_rate_labels, lsbp_labels, ldbp_labels, rsbp_labels, rdbp_labels, height_labels, weight_labels, \
                waist_labels, bmi_labels, exercise_freq_labels, smoking_status_labels, drinking_freq_labels, heart_rate_labels, tcho_labels,tg_labels,ldlc_labels,hdlc_labels,is_hyper_labels = \
                    input_ids.squeeze(), age_labels.squeeze(),body_tem_labels.squeeze(), pulse_rate_labels.squeeze(), breathing_rate_labels.squeeze(), lsbp_labels.squeeze(), ldbp_labels.squeeze(), \
                    rsbp_labels.squeeze(), rdbp_labels.squeeze(), height_labels.squeeze(), weight_labels.squeeze(), waist_labels.squeeze(), bmi_labels.squeeze(), exercise_freq_labels.squeeze(), \
                    smoking_status_labels.squeeze(), drinking_freq_labels.squeeze(), heart_rate_labels.squeeze(), tcho_labels.squeeze(),tg_labels.squeeze(),ldlc_labels.squeeze(),hdlc_labels.squeeze(),is_hyper_labels.squeeze(dim=0)

                with torch.no_grad():
                    loss, is_hyper_logits, bert_predict_tensor, is_hyper_labels = model(input_ids, age_labels=age_labels,body_tem_labels=body_tem_labels, pulse_rate_labels=pulse_rate_labels,breathing_rate_labels=breathing_rate_labels,
                                        lsbp_labels=lsbp_labels, ldbp_labels=ldbp_labels,rsbp_labels=rsbp_labels,rdbp_labels=rdbp_labels,
                                        height_labels=height_labels, weight_labels=weight_labels,waist_labels=waist_labels,bmi_labels=bmi_labels,
                                        exercise_freq_labels=exercise_freq_labels, smoking_status_labels=smoking_status_labels,drinking_freq_labels=drinking_freq_labels,
                                        heart_rate_labels=heart_rate_labels,tcho_labels=tcho_labels,tg_labels=tg_labels,ldlc_labels=ldlc_labels,hdlc_labels=hdlc_labels,is_hyper_labels=is_hyper_labels,epoch=global_step,flag=1)
                    print('bert_predict_tensor',bert_predict_tensor.shape)
                    print('is_hyper_labels',is_hyper_labels.shape)



                    y_preds.append(t2n(torch.sigmoid(is_hyper_logits)))
                    y_trues.append(t2n(is_hyper_labels))

                    # test_loss=0
                    test_loss_1 += loss.item()
                    test_loss_2 = loss.item()
                    # print(test_loss)
                    nb_tr_examples += 1
                    nb_tr_steps += 1

                    # Display loss
                    # test_loss_list.append('%.4f' % (test_loss/nb_tr_steps))
                    # loss_list_adv.append(loss.item)

            acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                          args.therhold)
            print(acc_container)
            # _, ax1 = plt.subplots()
            # ax2 = ax1.twinx()
            # x=y_preds[0].tolist()
            # print(x)

            # plt.plot(y_preds[0].tolist(), y_trues[0].tolist(),linestyle='-',marker='o')
            # plt.axis([0,1,0,1])

            # ax2.plot(np.arange(2543), y_trues[0].tolist())
            # plt.set_xlabel('y_preds')
            # plt.set_ylabel('y_trues')
            # plt.show()
            # ax2.set_ylabel('y_trues')

            # fitlog.add_loss(loss, name="Loss")

            # save report
            if args.do_train:
                for k, v in acc_container.items():
                    writer.add_scalar(
                        'test/{}'.format(k), v, 0)

            return acc_container

        test(task=0)


if __name__ == "__main__":
    main()
