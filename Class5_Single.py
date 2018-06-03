import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt


class Class5_Single(object):

    def __init__(self, config, learning_rate, range):
        self.config = config
        self.data = []
        self.train_acc = []
        self.test_acc = []
        self.load_csv()
        self.set_x_y()
        self.set_model(learning_rate, range)

    def load_csv(self):
        f = open('Dataset_Class5.csv', 'r')
        rdr = csv.reader(f)
        for line in rdr:
            start = False
            cnt = 0
            tmp = [int(i) for i in line]
            for temp in tmp:
                if start is False:
                    start = True
                    continue
                else:
                    if temp == 0:
                        cnt = cnt + 1
                    else:
                        cnt = cnt - 1
            if cnt == 24:
                start = False
                cnt = 0
                continue
            else:
                start = False
                cnt = 0
                self.data.append(tmp)
        f.close()

    def set_x_y(self):
        y_vals = []

        for val in self.data:
            temp = []
            for i in range(self.config.index):
                temp.append(0)
            temp[val[0]] = 1
            y_vals.append(temp)
            continue

        self.y_vals = np.array(y_vals)
        self.x_vals = np.array([val[1:25] for val in self.data])

        train_indices = np.random.choice(len(self.x_vals), round(len(self.x_vals)*0.7),replace=False)
        test_indices = np.array(list(set(range(len(self.x_vals))) - set(train_indices)))

        self.x_vals_train = self.x_vals[train_indices]
        self.x_vals_test = self.x_vals[test_indices]
        self.y_vals_train = self.y_vals[train_indices]
        self.y_vals_test = self.y_vals[test_indices]

    def set_model(self, learning_rate, range1):

        x = tf.placeholder(tf.float32, [None, 24])
        W = tf.Variable(tf.zeros([24,self.config.index]))
        b = tf.Variable(tf.zeros([self.config.index]))
        y = tf.nn.softmax(tf.matmul(x,W) + b)
        y_ = tf.placeholder(tf.float32, [None,self.config.index])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Single Layer")
        print("Index : "+str(self.config.index))
        print("Learning rate : "+ str(learning_rate))
        print("Range : " + str(range1))
        for i in range(range1):
            rand_index = np.random.choice(len(self.x_vals_train), size= self.config.batch_size)
            rand_x = self.x_vals_train[rand_index]
            rand_y = self.y_vals_train[rand_index]
            sess.run(train_step, feed_dict={x:rand_x, y_:rand_y})
            temp_loss = sess.run(cross_entropy, feed_dict={x:rand_x, y_:rand_y})
            if (i+1) % self.config.print_range == 0:
                print('#' + str(i+1) + ' Loss = ' + str(temp_loss))
            temp_acc_train = sess.run(accuracy, feed_dict={x: self.x_vals_train, y_: self.y_vals_train})
            self.train_acc.append(temp_acc_train)
            temp_acc_test = sess.run(accuracy, feed_dict={x: self.x_vals_test, y_: self.y_vals_test})
            self.test_acc.append(temp_acc_test)

        print(" Accuracy : " + str(sess.run(accuracy, feed_dict={x: self.x_vals_test, y_: self.y_vals_test})))
        plt.plot(self.train_acc, 'k-', label = 'Train Set Accuracy')
        plt.plot(self.test_acc, 'r--', label = 'Test Set Accuracy')
        plt.title('Train and Test Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig('Single_Index_'+str(self.config.index)+'_LearningRate_'+str(learning_rate)+"_Range_"+str(range1)+".jpg")
        plt.close()
        print("----------------------------------")
