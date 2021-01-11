import numpy as np
import tensorflow as tf
import argparse
import os
import moxing as mox

tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

mox.file.make_dirs('/cache/data')
mox.file.make_dirs('/cache/log')
mox.file.copy_parallel('obs://sh-ml-project/fabric/processed_data/task1/size256/1', '/cache/data')

# Training Parameters
learning_rate = 0.0005
num_steps = 2000

batch_size = 64
num_epochs = 10

# Network Parameters
num_input = 196608  # data input (img shape: 256*256*3)
num_classes = 3
dropout = 0.25  # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Convolution Layer with 128 filters and a kernel size of 5
        conv3 = tf.layers.conv2d(conv2, 128, 5, activation=tf.nn.relu, padding='same')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv3)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


def main():
    x_train = np.load("/cache/data/x_train.npy")
    y_train = np.load("/cache/data/y_train.npy")
    x_test = np.load("/cache/data/x_test.npy")
    y_test = np.load("/cache/data/y_test.npy")

    x_train = x_train * 1.0 / 127.5 - 1
    x_test = x_test * 1.0 / 127.5 - 1
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print("data load finish")

    print('Training model...')
    # Build the Estimator

    model = tf.estimator.Estimator(model_fn, model_dir='/cache/log/',
                                   config=tf.estimator.RunConfig(save_summary_steps=10, keep_checkpoint_max=1,
                                                                 log_step_count_steps=10))

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': x_train}, y=y_train,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model

    model.train(input_fn, steps=num_steps)
    print('Done training!')

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': x_test}, y=y_test,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    print('total accuracy: {}'.format(model.evaluate(input_fn)['accuracy']))

    for c in np.nditer(np.unique(y_test)):
        x_temp = np.array([x_test[i] for i in range(0, len(y_test)) if y_test[i] == c])
        y_temp = np.zeros((x_temp.shape[0],), dtype=np.int) + c
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': x_temp}, y=y_temp,
            batch_size=batch_size, shuffle=False)
        print('class{} accuracy: {}'.format(c, model.evaluate(input_fn)['accuracy']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', default='D://fabric_data_new/', type=str)
    parser.add_argument('--train_url', default='D://fabric_data_new/', type=str)
    parser.add_argument('--num_gpus', default=1, type=int)
    args = parser.parse_args()
    main()
    # mox.file.copy_parallel('/cache/log/eval', 'obs://sh-ml-project/fabric/log/task1/size256/4/eval')
    # paths = os.listdir('/cache/log')
    # print(paths)
    # for path in paths:
    #     if (path.find('data') != -1 or path == 'eval'):
    #         continue
    #     mox.file.copy('/cache/log/' + path, 'obs://sh-ml-project/fabric/log/task1/size256/4/' + path)
