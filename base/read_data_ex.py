# -*- coding:utf-8 -*-

import numpy as np 
import json

# draw the ui matrix
def draw_mat(rate):
    ui_mat = np.zeros([943,1682], dtype = np.float32)
    if(rate == '100'):
        with open('../data/u.data','r') as f:
            lines = f.readlines()
    else:
        with open('../data/data_train_{}'.format(str(rate)),'r') as f:
            lines = f.readlines()

    # first draw the ui matrix
    for line in lines:
        line_data = line.split('\t')
        ui_mat[int(line_data[0]) - 1,int(line_data[1]) - 1] = int(line_data[2])

    return ui_mat



# input mode and rate,return the data of user/userinfo/item/iteminfo as 2-d matrix
def read_data(mode,rate):

    ui_mat = draw_mat(rate)

    if(mode == 'train'):
        with open('../data/data_train_{}'.format(str(rate)),'r') as f:
            lines = f.readlines()
    elif(mode == 'rmse'):
        with open('../data/data_test_{}'.format(str(rate)),'r') as f:
            lines = f.readlines()

    user_data = np.zeros([len(lines),1682], dtype = np.float32)
    user_info = np.zeros([len(lines),29], dtype = np.float32)
    item_data = np.zeros([len(lines),943], dtype = np.float32)
    item_info = np.zeros([len(lines),26], dtype = np.float32)
    R = np.zeros([len(lines),1], dtype = np.float32)

    with open('../data/sideinfo.json','r') as f:
        json_text = f.read()

    sideinfo = json.loads(json_text)
    userinfo = np.array(sideinfo['user'], dtype = np.float32)
    iteminfo = np.array(sideinfo['item'], dtype = np.float32)

    index = 0
    for line in lines:
        line_data = line.split('\t')
        user_data[index,:] = ui_mat[int(line_data[0]) - 1, :]
        # print ui_mat[int(line_data[0]) - 1,:].shape
        user_info[index,:] = userinfo[int(line_data[0]) - 1, :]
        # print user_info[int(line_data[0]) - 1, :].shape
        item_data[index,:] = ui_mat[:, int(line_data[1]) - 1].T 
        # print ui_mat[:,int(line_data[1]) - 1].T.shape
        item_info[index,:] = iteminfo[int(line_data[1]) - 1, :]
        # print iteminfo[int(line_data[1]) - 1, :].shape
        R[index,0] = int(line_data[2])
        index += 1
    print(R)
    return user_data,user_info,item_data,item_info,R

# to calculate the recall,we need all the data
def read_data_recall():
    ui_mat = draw_mat('100')

    user_data = ui_mat
    item_data = ui_mat.T

    with open('../data/sideinfo.json','r') as f:
        json_text = f.read()

    sideinfo = json.loads(json_text)
    user_info = np.array(sideinfo['user'], dtype = np.float32)
    item_info = np.array(sideinfo['item'], dtype = np.float32)

    print(user_data.shape)
    print(user_info.shape)
    print(item_data.shape)
    print(item_info.shape)

    return user_data,user_info,item_data,item_info


# w1 and w2 is the vector of user and item, mutiple the to get the rate and sort them,
# and return the recall for K = 50,100,150,200,250,300
def get_recall(w1,w2):

    ui_mat = draw_mat('100') # shape = (943,1682)
    sorted_rate = []

    for i in range(943):
        w = w1[i, :].T.reshape(-1,1)  # shape = (20,1)
        rate = np.dot(w2,w) # shape = (1682,1)
        rate_list = []
        for j in range(1682):
            rate_list.append((j,rate[j,0]))  #[(0,0.2),(1,0.001),(3,0.981)...]
        rate_list = sorted(rate_list,key = lambda x:x[1],reverse = True)
        print(rate_list)

        sorted_rate.append(rate_list)

    sum_like = np.sum(ui_mat,axis = 1,keep_dim = True) # shape = (943,1)
    recall_dict = {}
    for K in [50,100,150,200,250,300]:
        topK_like = sorted_rate[:][0:K] #the top-K likes of all the users, shape = (943,K)
        sum_recall = 0
        for i in range(943):
            num_like = 0
            for t in topK_like[i]:
                if(ui_mat[i,t[1]] > 3):
                    num_like += 1
            recall = num_like * 1.0 / sum_like[i,0]
            sum_recall += recall 

        ave_racall = sum_recall / 943

        recall_dict[K] = ave_racall

    return recall_dict



