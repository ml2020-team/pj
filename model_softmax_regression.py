import argparse
import numpy as np
import tensorflow as tf
import moxing as mox

print(tf.__version__)

mox.file.make_dirs('/cache/data')
mox.file.make_dirs('/cache/log')
mox.file.copy_parallel('obs://sh-ml-project/fabric/processed_data/task1/size64/1', '/cache/data')

# 768 191
# 1609 191
# 1181 294
# 2706 294
# Parameters
train_size = 768
test_size = 191
num_class = 3
p_lamda = 0.5
learning_rate = 0.0005
training_epochs = 2000
batch_size = train_size
display_step = 1


def main():
    x_train = np.load("/cache/data/x_train.npy").reshape([train_size, 12288])
    y_train = np.load("/cache/data/y_train.npy")
    y_train = tf.keras.utils.to_categorical(y_train)

    x_test = np.load("/cache/data/x_test.npy").reshape([test_size, 12288])
    y_test = np.load("/cache/data/y_test.npy")
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train * 1.0 / 127.5 - 1
    x_test = x_test * 1.0 / 127.5 - 1
    print("data load finish")

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 12288])  # data image of shape 64*64*3=12288
    y = tf.placeholder(tf.float32, [None, num_class])

    # Set model weights
    W = tf.Variable(tf.zeros([12288, num_class]))
    b = tf.Variable(tf.zeros([num_class]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy and L2_regulation
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1)) + tf.contrib.layers.l2_regularizer(
        p_lamda)(W)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(x_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = x_train
                batch_ys = y_train
                # Fit training using batch data
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', default='D://fabric_data_new/', type=str)
    parser.add_argument('--train_url', default='D://fabric_data_new/', type=str)
    parser.add_argument('--num_gpus', default=1, type=int)
    args = parser.parse_args()
    main()
