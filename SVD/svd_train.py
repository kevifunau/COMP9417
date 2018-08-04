
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SVD:
    def __init__(self, train_X, test_X, vector_dimension = 15, steps = 20, gamma = 0.04, Lambda = 0.15, slow_rate = 0.07):

        self.train_X = np.array(train_X)
        self.test_X = np.array(test_X)
        self.vector_dimension = vector_dimension

        self.steps = steps
        self.gamma = gamma
        self.Lamdba = Lambda
        self.slow_rate = slow_rate

        self.rate_average = np.mean(self.train_X[:,2])

        #init variables
        self.item_bias = {}
        self.user_bias = {}
        self.q_item = {}
        self.p_user = {}
        self.y = {}

        self.item_2_user = {}
        self.user_2_item = {}

        #read training data
        for user_id, item_id, rate, time in self.train_X:

            self.item_2_user.setdefault(item_id, {})
            self.user_2_item.setdefault(user_id, {})
            self.item_2_user[item_id][user_id] = rate
            self.user_2_item[user_id][item_id] = rate
            self.init_variables(item_id, user_id, 'train')


    def init_variables(self, item_id, user_id, mode = 'train'):
            self.item_bias.setdefault(item_id, 0)
            self.user_bias.setdefault(user_id, 0)
            self.y.setdefault(item_id, np.zeros((self.vector_dimension, 1)))

            if mode == 'test':
                self.q_item.setdefault(item_id, np.zeros((self.vector_dimension, 1)))
                self.p_user.setdefault(user_id, np.zeros((self.vector_dimension, 1)))
            elif mode == 'train':
                self.q_item.setdefault(item_id, np.random.random((self.vector_dimension, 1)) * np.sqrt(self.vector_dimension) / 10)
                self.p_user.setdefault(user_id, np.random.random((self.vector_dimension, 1)) * np.sqrt(self.vector_dimension) / 10)


    def prediction(self, user_id, item_id, mode = 'test', model = 'sdv'):
        if mode == 'test':
            self.init_variables(item_id, user_id, 'test')
        if model == 'sdv':
            predict_rate = self.rate_average + self.item_bias[item_id] + self.user_bias[user_id] + np.sum(self.q_item[item_id] * self.p_user[user_id])
        elif model == 'sdv++':
            predict_rate = self.rate_average + self.item_bias[item_id] + self.user_bias[user_id] + np.sum(self.q_item[item_id] * (self.p_user[user_id] + self.impl))
        return max(min(predict_rate, 5), 1)


    def parameter_update(self, error, oringinal_value):
        return self.gamma * (error - self.Lamdba * oringinal_value)


    def train(self, model = 'sdv'):
        print('Using ' + model + ' to train.')
        for step in range(self.steps):
            print('Training epoch ' + str(step) + '...')
            square_error = 0.0

            shuffled_index = np.random.permutation(self.train_X.shape[0])

            for index in shuffled_index:
                user_id, item_id, rate, time = self.train_X[index]

                error = rate - self.prediction(user_id, item_id, 'train', model)
                square_error += error ** 2

                self.user_bias[user_id] += self.parameter_update(error, self.user_bias[user_id])
                self.item_bias[item_id] += self.parameter_update(error, self.item_bias[item_id])

                original_q_item = self.q_item[item_id]
                original_p_user = self.p_user[user_id]

                self.p_user[user_id] += self.parameter_update(error * original_q_item, original_p_user)

                if model == 'sdv':
                    self.q_item[item_id] += self.parameter_update(error * original_p_user, original_q_item)
                elif model == 'sdv++':
                    item_list = list(self.user_2_item[user_id].keys())
                    self.impl = np.sum([self.y[id] for id in item_list], axis=0) / np.sqrt(len(item_list))
                    self.q_item[item_id] += self.parameter_update(error * (original_p_user + self.impl), original_q_item)
                    for user_rated_item_id in item_list:
                        self.y[user_rated_item_id] += self.parameter_update(error * self.q_item[user_rated_item_id]/np.sqrt(len(item_list)), self.y[user_rated_item_id])


            self.gamma *= (1 - self.slow_rate)

            print("Train set RMSE:", np.sqrt(square_error / self.train_X.shape[0]))

            self.test()


    def test(self):

        square_error = 0

        for user_id, item_id, rate, time in self.test_X:
            square_error += (rate - self.prediction(user_id, item_id, 'test')) ** 2

        print("Test  set RMSE:",np.sqrt(square_error / self.test_X.shape[0]))



if __name__ == '__main__':
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names = header)

    train_data, test_data = train_test_split(df, test_size = 0.1)
    SVD_model = SVD(train_data, test_data)
    SVD_model.train('sdv')
