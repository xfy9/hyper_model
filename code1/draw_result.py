# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

# 获取txt字典数据并转为列表
def get_txt_dic(file_txt):
    file = open(file_txt,'r')
    list_jaccard = []
    list_f1 = []
    list_prauc = []
    list_avg_p = []
    list_avg_r = []
    with open(file_txt) as f:
        for line in f:
            #print(line)
            dic = eval(line)
            #print(dic)
            list_jaccard.append(dic['jaccard'])
            list_f1.append(dic['f1'])
            list_prauc.append(dic['prauc'])
            list_avg_p.append(dic['avg_p'])
            list_avg_r.append(dic['avg_r'])
    file.close()
    return list_jaccard,list_f1,list_prauc,list_avg_p,list_avg_r

# 画折线图
def draw_linear():
    x = []
    for i in range(1,101):
        x.append(i);
    loss_all = []
    loss_no_re = []
    loss_no_pre = []
    loss_simre = []
    loss_no_reward = []
    loss_no_store_pre = []
    with open('train_loss_all_tangniaobing.txt') as f:
        for line in f:
            line=line[0:len(line)-1]
            loss_all.append(float(line))
    # with open('train_loss_gaoxueya_no_re_tangniaobing.txt') as f:
    #     for line in f:
    #         line = line[0:len(line) - 1]
    #         loss_no_re.append(float(line))

    with open('train_loss_nopre_tangniaobing.txt') as f:
        for line in f:
            line = line[0:len(line) - 1]
            loss_no_pre.append(float(line))

    with open('train_loss_nostore_tangniaobing.txt') as f:
        for line in f:
            line = line[0:len(line) - 1]
            loss_simre.append(float(line))


    with open('train_loss_tangniaobing_no_store_pre_unbalanced.txt') as f:
        for line in f:
            line = line[0:len(line) - 1]
            loss_no_store_pre.append(float(line))

    # with open('train_loss_gaoxueya_no_store_balanced_wmz.txt') as f:
    #     for line in f:
    #         line = line[0:len(line) - 1]
    #         loss_simre.append(float(line))

    # with open('train_loss_gaoxueya_no_reward_balanced_wmz.txt') as f:
    #     for line in f:
    #         line = line[0:len(line) - 1]
    #         loss_no_reward.append(float(line))

    # print(len(loss_all))
    # print(len(loss_no_re))
    # print(len(loss_no_pre))

    plt.plot(x, loss_no_store_pre, color='blue', linestyle='-', label='CDR-Detector$_{pre-rep-}$')
    # plt.plot(x, loss_no_store_pre, color='blue', linestyle='-', label='CDR-Detector-pre-rep-')
    plt.plot(x, loss_simre, color='red', linestyle='-', label='CDR-Detector$_{rep-}$')
    plt.plot(x, loss_no_pre, color='green', linestyle='-', label='CDR-Detector$_{pre-}$')
    plt.plot(x, loss_all, color='pink', linestyle='-', label='CDR-Detector')
    # plt.plot(x, loss_no_re, color='blueviolet',  linestyle='-', label='pre-no-re')


    # plt.plot(x, loss_no_reward, color='blue', linestyle='-', label='CDR-Detector-no-reward')


    # plt.plot(x, loss_all, color='orangered', marker='o', linestyle='-', label='pre-re')
    # plt.plot(x, loss_no_re, color='blueviolet', marker='D', linestyle='-.', label='pre-nore')
    # plt.plot(x, loss_no_pre, color='green', marker='*', linestyle='-', label='pre-nopre')
    plt.legend()  # 显示图例
    # plt.xticks(x, names, rotation=45)
    plt.xlabel("num of iterations")  # X轴标签
    plt.ylabel("loss")  # Y轴标签
    plt.ylim(0, 0.2)
    plt.show()

# 画散点图
def draw_scatter(x,y,kind):
    # 画图
    plt.scatter(x, y, c='b',label=kind)
    plt.show()

# 这里导入你自己的数据
# ......
# ......
# x_axix，train_pn_dis这些都是长度相同的list()

# 开始画图

# python 一个折线图绘制多个曲线
def get_avg_result(list_jaccard, list_f1, list_prauc, list_avg_p, list_avg_r,a):
    avg_jacc_1 = avg_f1_1 = avg_prauc_1 = avg_p_1 = avg_r_1 = 0;
    for i in range(len(list_jaccard)):
        avg_jacc_1 += list_jaccard[i];
        avg_f1_1 += list_f1[i];
        avg_prauc_1 += list_prauc[i];
        avg_p_1 += list_avg_p[i];
        avg_r_1 += list_avg_r[i];
    avg_jacc_1 /= a;
    avg_f1_1 /= a
    avg_prauc_1 /= a
    avg_p_1 /= a
    avg_r_1 /= a

    print("avg_jacc_1 ",avg_jacc_1," avg_f1_1 ",avg_f1_1," avg_prauc_1 ",avg_prauc_1," avg_p_1 ",avg_p_1," avg_r_1 ",avg_r_1)



if __name__ == "__main__":

    # 画折线图
    draw_linear()

    #list_jaccard, list_f1, list_prauc, list_avg_p, list_avg_r =  get_txt_dic('result/test/2023-07-02/test_no_store_pre_unblance_tangniaobing.txt')
    # list_jaccard_train, list_f1_train, list_prauc_train, list_avg_p_train, list_avg_r_train = get_txt_dic('result/train/2022-10-27/junheng_train_stand.txt')
    #print(list_jaccard)
    #get_avg_result(list_jaccard,list_f1,list_prauc,list_avg_p,list_avg_r,5)
    # get_avg_result(list_jaccard_train,list_f1_train,list_prauc_train,list_avg_p_train,list_avg_r_train,5)

    # list_x_axix = []
    # for i in range(len(list_prauc)):
    #     list_x_axix.append(i+1)
    #
    # #  折线图
    # # sub_axix = filter(lambda x: x % 3 == 0, list_x_axix)
    # plt.title('Result Analysis')
    # #plt.scatter(list_x_axix, list_jaccard, color='green', label='jaccard')
    # plt.scatter(list_x_axix, list_f1, color='red', label='f1')
    # plt.scatter(list_x_axix, list_prauc, color='skyblue', label="prauc")
    # # # plt.plot(list_x_axix, list_avg_p, color='blue', label='avg_p')
    # # # plt.plot(list_x_axix, list_avg_r, color='yellow', label='avg_r')
    # plt.legend()  # 显示图例
    #
    # plt.xlabel('iteration times')
    # plt.ylabel('rate')
    # plt.show()

    #draw_scatter(x=list_x_axix,y=list_jaccard,kind="jaccard")
    #draw_scatter(x=list_x_axix, y=list_f1, kind="f1")
    #draw_scatter(x=list_x_axix, y=list_prauc, kind="prauc")




