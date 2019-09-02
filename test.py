import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.compat.v1 import placeholder
from tensorflow.compat.v1 import global_variables_initializer
from tensorflow.compat.v1.train import GradientDescentOptimizer
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_dataset():
    """
    Opens sonar.csv and extracts features and labels from the dataset
    :rtype: tuple
    :return: return the features and one hot encoded label
    """
    df = pd.read_csv("sonar.csv")
    x = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = one_hot_encode(y)
    return x, y


def one_hot_encode(labels):
    """
    One hot encodes the labels like for example if we have three categories 0, 1, 2 then
    the encoder will encode it as (1,0,0), (0,1,0), (0,0,1)
    :param labels: numerical labels that need to be one hot encoded
    :return: one hot encoded labels based on the number of labels and number of unique labels
    """
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode_output = np.zeros((n_labels, n_unique_labels))
    one_hot_encode_output[np.arange(n_labels)] = 1
    return one_hot_encode_output


def multilayer_perceptron(x, weights, biases):
    layer1 = tf.matmul(x, weights['h1'] + biases['b1'])
    layer1 = tf.nn.sigmoid(layer1)

    layer2 = tf.matmul(layer1, weights['h2'] + biases['b2'])
    layer2 = tf.nn.sigmoid(layer2)

    layer3 = tf.matmul(layer2, weights['h3'] + biases['b3'])
    layer3 = tf.nn.sigmoid(layer3)

    layer4 = tf.matmul(layer3, weights['h4'] + biases['b4'])
    layer4 = tf.nn.relu(layer4)

    output_layer = tf.matmul(layer4, weights['out'] + biases['out'])

    return output_layer


if __name__ == "__main__":
    X, Y = read_dataset()
    X, Y = shuffle(X, Y, random_state=45)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)
    print train_x.shape
    print train_y.shape
    print test_x.shape

    # define our hyper parameters
    learning_rate = 0.1
    training_epochs = 100
    cost_history = np.empty(shape=[1], dtype=float)
    n_dim = X.shape[1]
    print'n_dim: ' + str(n_dim)
    n_class = 2

    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60

    x = placeholder(tf.float32, [None, n_dim])
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))
    y_ = placeholder(tf.float32, [None, n_class])

    weights = {
        'h1': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_hidden_4, n_class]))
    }

    biases = {
        'b1': tf.Variable(tf.random.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.truncated_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random.truncated_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random.truncated_normal([n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_class]))
    }

    init = global_variables_initializer()
    y = multilayer_perceptron(x, weights, biases)

    cost_fct = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    training_step = GradientDescentOptimizer(learning_rate).minimize(cost_fct)

    with tf.Session() as sess:
        sess.run(init)

        # Mean squared error
        mse_history = []
        accuracy_history = []

        for epoch in range(training_epochs):
            sess.run(training_step, feed_dict={x: train_x, y_: train_y})
            cost = sess.run(cost_fct, feed_dict={x: train_x, y_: train_y})
            cost_history = np.append(cost_history, cost)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            print(tf.cast(correct_prediction, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            pred_y = sess.run(y, feed_dict={x: test_x})
            mse = tf.reduce_mean(tf.square(pred_y - test_y))
            mse_ = sess.run(mse)
            mse_history.append(mse_)
            accuracy = (sess.run(accuracy, feed_dict={x:train_x, y_:train_y}))
            accuracy_history.append(accuracy)
            print('epoch: ', epoch)
            print('correct prediction: ', correct_prediction)
            print('cost: ', cost)
            print('MSE: ', mse_)
            print('Training Accuracy: ', accuracy)

        plt.plot(mse_history, "r")
        plt.show()
        plt.plot(accuracy_history)
        plt.show()

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x:test_x, y_:test_y})))
        pred_y = sess.run(y, feed_dict={x: test_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        print('MSE: %4f' % sess.run(mse))