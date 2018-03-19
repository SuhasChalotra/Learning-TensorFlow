import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from  sklearn.metrics import confusion_matrix
import time
from datetime import timedelta


#Import linear_model to get some functions



filter_size1 = 5
num_filter1 = 16
filter_size2 = 5
num_filter2 = 36
fc_size = 128
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)


img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10


def new_conv_layer(input, num_input_channels, filter_size, num_filters,use_pooling=True):

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
    layer = tf.nn.conv2d(input=input, filter=weights, strides= [1,1,1,1], padding='SAME')

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs,use_relu=True):

    weights = tf.Variable(tf.truncated_normal(shape=[num_inputs, num_outputs]))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        return tf.nn.relu(layer)
    else:
        return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels,

                                            filter_size=filter_size1, num_filters=num_filter1)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,num_input_channels=num_filter1,
                                            filter_size=filter_size2,num_filters=num_filter2)

layer_flat , num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size)
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

correct_pred = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0

def optimize(num_iterations):
    global total_iterations

    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        feed_dict_train = {x:x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i%100==0:
            training_accuracy = session.run(accuracy,feed_dict_train)
            print("Iteration num: {0:>6}, Training acc: {1:>6.1%}".format(i, training_accuracy))

    total_iterations += num_iterations
    end_time = time.time()

    print("Time usage:{0}".format(timedelta(seconds=int(round(end_time - start_time)))))


# Helper functions to visualize performance
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)

    images = data.test.images[incorrect]

    cls_pred = cls_pred[incorrect]

    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


# Split the test-set into smaller batches of this size.
test_batch_size = 256


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test

    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

#This should tell us our untrained accuracy
print_test_accuracy()

optimize(400)
print_test_accuracy(show_confusion_matrix=True)
session.close()





