# -*- coding:utf-8 -*-

# this file is aiming to pre-process the side infomation
import json

sideinfor = {}
info_user = []
# zipcode_dict = {}
occu_dict = {'administrator':0,
'artist':1,
'doctor':2,
'educator':3,
'engineer':4,
'entertainment':5,
'executive':6,
'healthcare':7,
'homemaker':8,
'lawyer':9,
'librarian':10,
'marketing':11,
'none':12,
'other':13,
'programmer':14,
'retired':15,
'salesman':16,
'scientist':17,
'student':18,
'technician':19,
'writer':20}

with open("../data/u.user","r") as f:
    lines = f.readlines()
    # print len(lines)
    # for line in lines:
    #     info = line.split('|')
    #     zipcode = info[4].split('\n')[0]
    #     if(zipcode not in zipcode_dict.keys()):
    #         zipcode_dict[zipcode] = len(zipcode_dict)

    # zipcode_len = len(zipcode_dict)
    # print zipcode_dict
    # print zipcode_len #795

    for line in lines:
        info_line = [0] * 29
        info = line.split('|')
        age = int(info[1])
        if(age < 18):
            info_line[0] = 1
        elif(age < 25):
            info_line[1] = 1
        elif(age < 35):
            info_line[2] = 1
        elif(age < 45):
            info_line[3] = 1
        elif(age < 50):
            info_line[4] = 1
        elif(age < 55):
            info_line[5] = 1
        else:
            info_line[6] = 1

        gender = info[2]
        if(gender == 'M'):
            info_line[7] = 1

        occupation = info[3]
        if(occupation not in occu_dict.keys()):
            print 'occupation error'
            print info[0]
            break
        else:
            info_line[8 + occu_dict[occupation]] = 1

        # zipcode = info[4].split('\n')[0]

        # if(zipcode not in zipcode_dict.keys()):
        #     print 'zipcode error'
        #     print info[0]
        #     break
        # else:
        #     # print zipcode_dict[zipcode]
        #     info_line[29 + zipcode_dict[zipcode]] = 1

        info_user.append(info_line)

# print info_user[23]
sideinfor['user'] = info_user

info_item = []
date_dict = {}
with open('../data/u.item','r') as f:
    lines = f.readlines()
    for line in lines:
        info_line = [0] * 26
        info = line.split('|')
        date = info[2].split('-')
        # print date
        if(len(date) == 3):
            year = int(date[2])
        else:
            year = 0
        
        if(year > 1995):
            info_line[0] = 1
        elif(year > 1989):
            info_line[1] = 1
        elif(year > 1979):
            info_line[2] = 1
        elif(year > 1969):
            info_line[3] = 1
        elif(year > 1959):
            info_line[4] = 1
        elif(year > 1949):
            info_line[5] = 1
        else:
            info_line[6] = 1

        
        for i in range(19):
            info_line[7 + i] = int(info[5 + i])

        info_item.append(info_line)


# print info_item[35]
sideinfor['item'] = info_item

with open('../data/sideinfo.json', 'w') as f:
    js = json.dumps(sideinfor)
    f.write(js)
    

with open('../data/u.data','r') as f:
    lines = f.readlines()
    total_size = len(lines)
    
    cut = int(total_size * 0.6)

    with open('../data/data_train_60','w') as fw:
        for i in range(cut):
            fw.write(lines[i])

    with open('../data/data_test_60','w') as fw:
        for i in range(cut,total_size):
            fw.write(lines[i])

    cut = int(total_size * 0.8)

    with open('../data/data_train_80','w') as fw:
        for i in range(cut):
            fw.write(lines[i])

    with open('../data/data_test_80','w') as fw:
        for i in range(cut,total_size):
            fw.write(lines[i])


    cut = int(total_size * 0.95)

    with open('../data/data_train_95','w') as fw:
        for i in range(cut):
            fw.write(lines[i])

    with open('../data/data_test_95','w') as fw:
        for i in range(cut,total_size):
            fw.write(lines[i])
