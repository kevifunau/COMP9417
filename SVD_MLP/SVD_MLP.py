# COMP9417 PROJECT RECOMMONDER SYSTEM
# Python3.6
# Team memember:
# Hao Huang z5112059
# Kai Fang
# Dan Liu
# Yihao Wu
#
# Date: 30 May 2018
#==============================================#

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


#=============parameters definition=============#

# embedding_size: the length of the hidden vector for both user and item
emdedding_size_mlp = 20
emdedding_size_svd = 20

# batch_size: the number of cases put into our model for one term
batch_size = 100

# num_epoch: the total round we train our model
num_epoch = 100

# learing_rate: the variable chaging speed for our optimizer
learning_rate = 0.001


#==========Data reading and preprocessing========#

header = ['user_id', 'item_id', 'rating', 'timestamp']

# If we choose to use ml_100k dataset, we use the following way to read the data.
# df = pd.read_csv('ml_100k.data', sep='\t', names = header)
# n_users = df.user_id.unique().shape[0]
# n_items = df.item_id.unique().shape[0]


# If we choose to use ml_1m dataset, we use the following way to read the data.
df = pd.read_csv('ml_1m.data', sep='::', names = header)
n_users = max(df.user_id)
n_items = max(df.item_id)


# split our whole data into train_data and test_data, and caculate the overall average
train_data, test_data = train_test_split(df, test_size = 0.1)
train_data = np.array(train_data)
test_data = np.array(test_data)
rate_average = np.mean(train_data[:,2])


#================Model Definition================#


#----------------Initialize Part-----------------#
# we define our placeholders as the entrance of our data.
# keep_prob is for the drop_out in the full_connect layer.
user_id = tf.placeholder(tf.int32,[batch_size])
item_id = tf.placeholder(tf.int32,[batch_size])
rate = tf.placeholder(tf.float32,[batch_size,1])
keep_prob = tf.placeholder(tf.float32)


# We initialize our mlp user and item embedding matrix randomly here.
# As our user id sart from 1 so the shape is [n_users + 1, emdedding_size], the same for items.
user_embedding_mlp = tf.Variable(tf.random_uniform([n_users + 1, emdedding_size_mlp],0,0.3),trainable = True)
item_embedding_mlp = tf.Variable(tf.random_uniform([n_items + 1, emdedding_size_mlp],0,0.3),trainable = True)


# We also initialize our svd user and item embedding matrix randomly here.
# As our user id sart from 1 so the shape is [n_users + 1, emdedding_size], the same for items.
user_embedding_svd = tf.Variable(tf.random_uniform([n_users + 1, emdedding_size_svd],0,0.3),trainable = True)
item_embedding_svd = tf.Variable(tf.random_uniform([n_items + 1, emdedding_size_svd],0,0.3),trainable = True)

# We initialize our svd user bias and item bias embedding matrix zero here.
# As our user id sart from 1 so the shape is [n_users + 1, 1], the same for items.
user_bias_embedding = tf.Variable(tf.zeros([n_users + 1, 1]),trainable = True)
item_bias_embedding = tf.Variable(tf.zeros([n_items + 1, 1]),trainable = True)


#---------------------MLP Part-------------------#

# We use embedding_lookup to find out the item and user we are training or testing now for the mlp layer,
# both of them are with shape [bath_size, embedding_size_mlp].
user_input_mlp = tf.nn.embedding_lookup(user_embedding_mlp, user_id)
item_input_mlp = tf.nn.embedding_lookup(item_embedding_mlp, item_id)

# Here is the regularization term using l2_regularizer fot the user_input_mlp and item_input_mlp. We will include this term in the loss function for regularization.
regularizer_1 = tf.contrib.layers.l2_regularizer(0.3)
reg_term_1 = tf.contrib.layers.apply_regularization(regularizer_1,[user_input_mlp,item_input_mlp])

# We multiply the user hidden layer with the item hidden layer to get their combination with shape [bath_size, embedding_size_mlp]
fc_input = tf.concat([user_input_mlp,item_input_mlp],axis=1)

# Then we put the combined vector into our fc layers, with an out put fc_output.
# The shape of the fc_output is [batch_size, 1], which indicates the nonlinear connection between the item and the user.
w_1 = tf.Variable(tf.truncated_normal([2 * emdedding_size_mlp, 100], stddev=0.1))
b_1 = tf.Variable(tf.constant(0., shape=[100]))
l_1 = tf.nn.tanh(tf.nn.xw_plus_b(fc_input, w_1, b_1))
l_1_drop = tf.nn.dropout(l_1, keep_prob)

w_2 = tf.Variable(tf.truncated_normal([100, 20], stddev=0.1))
b_2 = tf.Variable(tf.constant(0., shape=[20]))
l_2 = tf.nn.tanh(tf.nn.xw_plus_b(l_1_drop, w_2, b_2))
l_2_drop = tf.nn.dropout(l_2, keep_prob)


w_4 = tf.Variable(tf.truncated_normal([20, 1], stddev=0.1))
b_4 = tf.Variable(tf.constant(0., shape=[1]))
fc_output = tf.nn.xw_plus_b(l_2_drop,w_4,b_4)


#---------------------SVD Part-------------------#

# We use embedding_lookup to find out the item and user we are training or testing now for the svd layer,
# both of them are with shape [bath_size, embedding_size_svd].
user_input_svd = tf.nn.embedding_lookup(user_embedding_svd, user_id)
item_input_svd = tf.nn.embedding_lookup(item_embedding_svd, item_id)


# Here is the regularization term using l2_regularizer fot the user_input_svd and item_input_svd. We will include this term in the loss function for regularization.
regularizer_2 = tf.contrib.layers.l2_regularizer(0.002)
reg_term_2 = tf.contrib.layers.apply_regularization(regularizer_2,[user_input_svd,item_input_svd])


# We use embedding_lookup to find out the bias of the item and user we are training or testing now for the svd layer,
# both of them are with shape [bath_size, 1].
user_bias = tf.nn.embedding_lookup(user_bias_embedding, user_id)
item_bias = tf.nn.embedding_lookup(item_bias_embedding, item_id)


# Here is the regularization term using l2_regularizer fot the user_bias and item_bias. We will include this term in the loss function for regularization.
regularizer_l1 = tf.contrib.layers.l1_regularizer(0.0005)
reg_term_3 = tf.contrib.layers.apply_regularization(regularizer_l1,[user_bias,item_bias])

# We calculate the qp term in the SVD.
svd_qp = tf.reduce_sum(tf.multiply(user_input_svd,item_input_svd), axis = 1, keep_dims = True)

# We get both ouput of the mlp and svd layer, then combine then into a new vector with shape [batch_size, 2].
prediction_mlp = fc_output
prediction_svd = svd_qp + user_bias + item_bias + rate_average
combine_vec = tf.concat([prediction_mlp, prediction_svd],axis = 1)

# Put the combined vector into a 1 layer fc layer for the final prediction.
w_3 = tf.Variable(tf.truncated_normal([2, 1], stddev=0.1))
b_3 = tf.Variable(tf.constant(0., shape=[1]))
prediction = tf.nn.xw_plus_b(combine_vec,w_3,b_3)

# We use mean_squared_error as the loss, and plus the regularization term.
loss = tf.losses.mean_squared_error(rate, prediction) + reg_term_1 + reg_term_2 + reg_term_3
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#===============Training our Model================#
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start learning...')
        for epoch in range(100):
            print('epoch: ',format(epoch))
            train_loss = []
            for start,end in zip(range(0,len(train_data), batch_size),range(batch_size,len(train_data),batch_size)):
                tr_loss,_ = sess.run([loss, train], feed_dict={user_id : train_data[start:end,0], item_id : train_data[start:end,1], rate: train_data[start:end,2].reshape(batch_size,1), keep_prob: 0.5})
                train_loss.append(tr_loss)


            rmse = []
            for start, end in zip(range(0, len(test_data), batch_size),range(batch_size, len(test_data), batch_size)):
                pred = sess.run(prediction, feed_dict={user_id : test_data[start:end,0], item_id : test_data[start:end,1], rate: test_data[start:end,2].reshape(batch_size,1), keep_prob: 1.0})
                pred = [[min(max(i[0],1),5)] for i in pred]
                pred = np.array(pred)
                rmse.append((pred - test_data[start:end,2].reshape(batch_size,1))**2)

            print("Train_rmse: {:.3f}, Test_rmse: {:.3f}".format(np.sqrt(np.mean(train_loss)), np.sqrt(np.mean(rmse))))




