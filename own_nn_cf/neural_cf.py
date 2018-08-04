import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


emdedding_size = 10
batch_size = 200


header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names = header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

train_data, test_data = train_test_split(df, test_size = 0.2)
train_data = np.array(train_data)
test_data = np.array(test_data)

user_id = tf.placeholder(tf.int32,[batch_size])
item_id = tf.placeholder(tf.int32,[batch_size])
rate = tf.placeholder(tf.float32,[batch_size,1])
keep_prob = tf.placeholder(tf.float32)


user_embedding = tf.Variable(tf.random_uniform([n_users + 1, emdedding_size]),trainable = True)
item_embedding = tf.Variable(tf.random_uniform([n_items + 1, emdedding_size]),trainable = True)

user_input = tf.nn.embedding_lookup(user_embedding, user_id)
item_input = tf.nn.embedding_lookup(item_embedding, item_id)

fc_input = tf.multiply(user_input,item_input)
w_1 = tf.Variable(tf.truncated_normal([emdedding_size, 100], stddev=0.1))
b_1 = tf.Variable(tf.constant(0., shape=[100]))
l_1 = tf.nn.relu(tf.nn.xw_plus_b(fc_input, w_1, b_1))
l_1_drop = tf.nn.dropout(l_1, keep_prob)


w_2 = tf.Variable(tf.truncated_normal([100, 50], stddev=0.1))
b_2 = tf.Variable(tf.constant(0., shape=[50]))
l_2 = tf.nn.relu(tf.nn.xw_plus_b(l_1_drop, w_2, b_2))
l_2_drop = tf.nn.dropout(l_2, keep_prob)

w_3 = tf.Variable(tf.truncated_normal([50, 1], stddev=0.1))
b_3 = tf.Variable(tf.constant(0., shape=[1]))
prediction = tf.nn.relu(tf.nn.xw_plus_b(l_2_drop, w_3, b_3))

loss = tf.losses.mean_squared_error(rate,prediction)
train = tf.train.AdamOptimizer(0.001).minimize(loss)


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start learning...')
        for epoch in range(100):
            print('epoch: ',format(epoch))
            train_loss = []
            for start,end in zip(range(0,len(train_data), batch_size),range(batch_size,len(train_data),batch_size)):
                #print(sess.run(fc_input,feed_dict={user_id : train_data[start:end,0], item_id : train_data[start:end,1], rate: train_data[start:end,2], keep_prob: 0.5}))
                tr_loss,_ = sess.run([loss, train], feed_dict={user_id : train_data[start:end,0], item_id : train_data[start:end,1], rate: train_data[start:end,2].reshape(batch_size,1), keep_prob: 0.5})
                train_loss.append(tr_loss)
            # Testing

            rmse = []
            for start, end in zip(range(0, len(test_data), batch_size),range(batch_size, len(test_data), batch_size)):
                pred = sess.run(prediction, feed_dict={user_id : test_data[start:end,0], item_id : test_data[start:end,1], keep_prob: 1.0})
                pred = [[min(max(i[0],1),5)] for i in pred]
                pred = np.array(pred)
                rmse.append((pred - test_data[start:end,2].reshape(batch_size,1))**2)


            print("loss: {:.3f}, rmse: {:.3f}".format(np.mean(train_loss), np.sqrt(np.mean(rmse))))
