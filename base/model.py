# -*- coding:utf-8 -*-

import tensorflow as tf
from read_data import read_data,read_data_recall,get_recall,read_data_recall_test,get_recall_test
import numpy as np
import json


hidden_size = 400   # the size of hidden layers
latent_size = 20    # the size of latent factors of user network and item network
beta = 0.05         # the coefficient of the KL divergence


# x,a gennerate w,z,z_prime
def full_connect(x,layersize):
    temp1 = tf.layers.dense(x,layersize[0],tf.nn.relu,kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001))
    output = tf.layers.dense(temp1,layersize[1],kernel_regularizer = tf.contrib.layers.l2_regularizer(0.001))

    return output


# x,a gennerate w,z,z_prime
def build_encode(x,a):
    mu_x = full_connect(x,[hidden_size,latent_size])
    sigma_x = full_connect(x,[hidden_size,latent_size])
    mu_a = full_connect(a,[latent_size,latent_size])
    sigma_a = full_connect(a,[latent_size,latent_size])
    mu = mu_x + mu_a
    sigma = sigma_x + sigma_a
    epsilon = tf.random_normal(tf.shape(sigma))
    z = mu + tf.exp(sigma) * epsilon

    return mu,sigma,z



# a gennerate mu,sigma
def build_encode_a(x):    
    mu = full_connect(x,[latent_size,latent_size])
    sigma = full_connect(x,[latent_size,latent_size])

    return mu,sigma



# input x,a; return loss function,output_size is a 2d vector [ui,sideinfo]
def build_foward(x,a,output_size):

    mu_w,sigma_w,w = build_encode(x,a)
    mu_z,sigma_z,z = build_encode(x,a)
    mu_z_prime,sigma_z_prime,z_prime = build_encode(x,a)
    mu_a,sigma_a = build_encode_a(a)
    reconstruct_x_w = full_connect(w,[hidden_size,output_size[0]])
    reconstruct_x_z = full_connect(w,[hidden_size,output_size[0]])
    reconstruct_x = reconstruct_x_w + reconstruct_x_z
    reconstruct_a = full_connect(w,[latent_size,output_size[1]])

    entropy1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = reconstruct_x, labels = x), reduction_indices = 1)
    entropy2 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = reconstruct_a, labels = a), reduction_indices = 1)
    
    # KL(q(z|x,a) || p(z))
    KL1 = beta * tf.reduce_sum(tf.exp(sigma_z) + tf.pow(mu_z, 2) - sigma_z - 1, reduction_indices = 1)
    # # KL(q(z'|x,a) || p(z'))
    KL2 = beta * tf.reduce_sum(tf.exp(sigma_z_prime) + tf.pow(mu_z_prime, 2) - sigma_z_prime - 1, reduction_indices = 1)
    # # KL(q(w|x,a) || p(w|a))
    KL3 = beta * tf.reduce_sum(sigma_a - sigma_w - 1 + tf.exp(sigma_w) / tf.exp(sigma_a) + tf.pow(mu_a - mu_w,2) / tf.exp(sigma_a) , reduction_indices = 1)

    loss = tf.reduce_mean(entropy1+entropy2+KL1+KL2+KL3)

    return loss,w

# build the total loss, including the loss of the two NN and the RMSE of Rij (conjuncted by addition)
def build_loss(x_user,a_user,x_item,a_item,R,):
    # for the training of user
    # layersize1 = [[20,20],[400,1682],[20,29]]
    # for the training of item
    # layersize2 = [[20,20],[400,943],[20,26]]

    loss_user,w_user = build_foward(x_user,a_user,[1682,29])
    loss_item,w_item = build_foward(x_item,a_item,[943,26])

    rate = tf.reduce_sum(w_user * w_item, reduction_indices = 1, keep_dims = True)

    RMSD_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = rate, labels = R))
    rate_sigmoid = tf.nn.sigmoid(rate)    
    RMSD = tf.sqrt(tf.reduce_mean(tf.pow(rate_sigmoid - R, 2)))
    # RMSD = tf.sqrt(tf.reduce_mean(tf.pow(rate - R, 2)))
    total_loss = loss_user + loss_item + RMSD

    return total_loss,rate_sigmoid,RMSD



def model(mode,rate,checkpoint = '999',K = 50):
    if(mode == 'train'):
        # train user
        with tf.device('/cpu:0'):
            global_step = tf.Variable(0,trainable=False)
            boundaries = [300,1000,2000,3000,4000]  
            learning_rates = [0.004,0.0005,0.00003,0.000003,0.0000003,0.00000003]  
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)
            # learning_rate = 0.005
            summary = ''
        with tf.device('/cpu:0'):
            num_epoch = 5000
            batch_size = 512
            user_data,user_info,item_data,item_info,r_data = read_data('train',rate)
            user_data_test,user_info_test,item_data_test,item_info_test,r_data_test = read_data('rmse',rate)
            user_data_r,user_info_r,item_data_r,item_info_r = read_data_recall_test()
            x_user = tf.placeholder(tf.float32, shape = [None,1682])
            a_user = tf.placeholder(tf.float32, shape = [None,29])
            x_item = tf.placeholder(tf.float32, shape = [None,943])
            a_item = tf.placeholder(tf.float32, shape = [None,26])
            R = tf.placeholder(tf.float32, shape = [None,1])
            # learning_rate = tf.train.exponential_decay(0.0005,global_step,1000,0.1,staircase=False)
            _,w_user = build_foward(x_user,a_user,[1682,29])
            _,w_item = build_foward(x_item,a_item,[943,26])

            total_loss,rate,RMSD = build_loss(x_user, a_user, x_item, a_item, R)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step)

            # train user
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(num_epoch):
                    # modified here
                    loss = 0
                    m = user_data.shape[0]
                    if(m % batch_size == 0):
                        num_batches = int(m / batch_size)
                    else:
                        num_batches = int(m / batch_size) + 1

                    for i in range(num_batches):
                        if(i == num_batches - 1):
                            batch_x_user = user_data[i * batch_size : m, :]
                            batch_a_user = user_info[i * batch_size : m, :]
                            batch_x_item = item_data[i * batch_size : m, :]
                            batch_a_item = item_info[i * batch_size : m, :]
                            batch_r = r_data[i * batch_size : m, :]
                        else:
                            batch_x_user = user_data[i * batch_size : (i + 1) * batch_size, :]
                            batch_a_user = user_info[i * batch_size : (i + 1) * batch_size, :]
                            batch_x_item = item_data[i * batch_size : (i + 1) * batch_size, :]
                            batch_a_item = item_info[i * batch_size : (i + 1) * batch_size, :]
                            batch_r = r_data[i * batch_size : (i + 1) * batch_size, :]
                        
                        data_dict = {}
                        data_dict[x_user] = batch_x_user
                        data_dict[a_user] = batch_a_user
                        data_dict[x_item] = batch_x_item
                        data_dict[a_item] = batch_a_item
                        data_dict[R] = batch_r
                        _, l,rmse_train = sess.run([optimizer,total_loss,RMSD], feed_dict = data_dict)

                    loss += l
                    # print(epoch)
                    if((epoch % 10 == 0) or (epoch == 4999)):
                        # calculate rmsd on test set
                        data_dict = {}
                        data_dict[x_user] = user_data_test
                        data_dict[a_user] = user_info_test
                        data_dict[x_item] = item_data_test
                        data_dict[a_item] = item_info_test
                        data_dict[R] = r_data_test
                        rate_test,rmsd = sess.run([rate,RMSD], feed_dict = data_dict)
                        
                        # calculate recall
                        
                        data_dict = {}
                        data_dict[x_user] = user_data_r
                        data_dict[a_user] = user_info_r
                        data_dict[x_item] = item_data_r
                        data_dict[a_item] = item_info_r                        
                        w1 = sess.run(w_user, feed_dict = data_dict)
                        w2 = sess.run(w_item, feed_dict = data_dict)

                        # recall is a dictionary, with K as key and recall as value
                        recall = get_recall_test(w1,w2)

                        # output result
                        string = 'after epoch '+ str(epoch) +', Loss:' + str(loss) +', training rmse: ' + str(rmse_train)+ ', rmsd:' + str(rmsd) +'\n'
                        string += str(recall)
                        string += '\n'
                        string += str(rate_test)
                        print(string)
                        summary += (string + '\n')
                        if(epoch > 2000):
                            saver=tf.train.Saver(max_to_keep=1)
                            saver.save(sess,'../data/summary_{}/model_{}_{}.ckpt'.format(rate,rate,epoch))
        with tf.device('/cpu:0'):            
            with open('../data/summary_{}/summary_{}_rmse'.format(rate,rate),'w') as f:
                f.write(summary)




    elif(mode == 'rmsd'):
        with tf.device('/cpu:0'):
            user_data,user_info,item_data,item_info,r_data = read_data('rmse',rate)
            x_user = tf.placeholder(tf.float32, shape = [None,1682])
            a_user = tf.placeholder(tf.float32, shape = [None,29])
            x_item = tf.placeholder(tf.float32, shape = [None,943])
            a_item = tf.placeholder(tf.float32, shape = [None,26])
            R = tf.placeholder(tf.float32, shape = [None,1])

            _,_,RMSD = build_loss(x_user, a_user, x_item, a_item, R)
            
            # train user
            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, "../data/summary_{}/model_{}_{}.ckpt".format(str(rate),str(rate),str(checkpoint)))
                data_dict = {}
                data_dict[x_user] = user_data
                data_dict[a_user] = user_info
                data_dict[x_item] = item_data
                data_dict[a_item] = item_info
                data_dict[R] = r_data
                loss = sess.run(RMSD, feed_dict = data_dict)

                string = 'RMSD is: ' + str(loss)
                print(string)


    # use build_forward to gennerate the vector of user and item, and pass them a function in read_data to get recall
    elif(mode == 'recall'):
        with tf.device('/cpu:0'):
            user_data,user_info,item_data,item_info = read_data_recall()
            x_user = tf.placeholder(tf.float32, shape = [None,1682])
            a_user = tf.placeholder(tf.float32, shape = [None,29])
            x_item = tf.placeholder(tf.float32, shape = [None,943])
            a_item = tf.placeholder(tf.float32, shape = [None,26])

            _,w_user = build_foward(x_user,a_user,[1682,29])
            _,w_item = build_foward(x_item,a_item,[943,26])

            with tf.Session() as sess:
                saver = tf.train.Saver()
                saver.restore(sess, "../data/summary_{}/model_{}_{}.ckpt".format(str(rate),str(rate),str(checkpoint)))
                data_dict = {}
                data_dict[x_user] = user_data
                data_dict[a_user] = user_info
                data_dict[x_item] = item_data
                data_dict[a_item] = item_info
                w1 = sess.run(w_user, feed_dict = data_dict)
                w2 = sess.run(w_item, feed_dict = data_dict)

                w1_np=w1.eval(session = sess)
                w2_np=w2.eval(session = sess)

                # recall is a dictionary, with K as key and recall as value
                recall = get_recall(w1_np,w2_np)
                with open('../data/summary_{}/recall_{}.json'.format(str(rate),str(rate)), 'w') as f:
                    js = json.dumps(recall, indent=4, separators=(',', ':'))
                    f.write(js)