import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST", one_hot=True)

print("Size of:")
print("- training data:\t\t{}".format(len(data.train.labels)))
print("- test data:\t\t{}".format(len(data.test.labels)))
print("- valid data:\t\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
print(data.test.cls[0:20])

img_size = 28
img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

num_classes = 10

# Now we design the neural network

# A helper function to print the images since they are flattened
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# First the input

x = tf.placeholder(tf.float32, [None, img_size_flat])

# this is the result we want after out neural network
# "processes" the input x
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])


#make the model

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.add(tf.matmul(x,weights),biases) # XW + B

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Setting the cost and optimizer


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_pred = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

session = tf.Session()

session.run(tf.initialize_all_variables())

batch_size = 100

def optimize(num_iter):
    for i in range(num_iter):
        x_batch, y_true_batch = data.train.next_batch(batch_size)

        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i%100==0:
            print("{0} iterations left".format(num_iter-i))


feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}

def print_accuracy():
    acc = session.run(accuracy,feed_dict=feed_dict_test)

    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    cls_true = data.test.cls
    cls_pred = session.run(y_pred_cls, feed_dict = feed_dict_test)

    matrix = confusion_matrix(y_true=cls_true,
                              y_pred=cls_pred)
    print(matrix)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors():
    correct, cls_pred = session.run([correct_pred, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_weights():
    w = session.run(weights)

    min_weight = np.min(w)
    max_weight = np.max(w)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):

        if i<10:

            image = w[:, i].reshape(img_shape)

            ax.set_xlabel("Weights: {0}".format(i))

            ax.imshow(image, vmin=min_weight, vmax=max_weight, cmap='seismic')


        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

#Untrained model
print_accuracy()
plot_example_errors()

#train with 1000 iterations
optimize(num_iter=1000)
plot_example_errors()
print_accuracy()
plot_weights()


