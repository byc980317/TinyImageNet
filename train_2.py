# coding: utf-8

# ### Tiny Imagenet Visual Recognition Challenge
#
# Tiny Imagenet has 200 Classes, each class has 500 traininig images, 50 Validation Images and 50 test images. Label Classes and Bounding Boxes are provided. More details can be found at https://tiny-imagenet.herokuapp.com/
#
# This challenge is part of Stanford Class CS 213N

# In[5]:

import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from AlexNet import AlexNet
from sklearn import preprocessing

IMAGE_DIRECTORY = './tiny-imagenet-200'
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 9832
VAL_IMAGES_DIR = './tiny-imagenet-200/val'
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS

def load_training_images(image_dir, batch_size=500):
    image_index = 0

    images = np.ndarray(shape=(NUM_IMAGES, 64, 64, 3))
    names = []
    labels = []

    print("Loading training images from ", image_dir)
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        if os.path.isdir(image_dir + type + '/images/'):
            type_images = os.listdir(image_dir + type + '/images/')
            # Loop through all the images of a type directory
            batch_index = 0
            # print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file)
                # print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :,:,:] = image_data

                    labels.append(type)
                    names.append(image)

                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break

    print("Loaded Training Images", image_index)
    return (images, np.asarray(labels), np.asarray(names))

def label_dict(val_data):
    return dict(zip(val_data.loc[:,'File'],val_data.loc[:,'Class']))

def load_validation_images(testdir, validation_data,label_dict,batch_size=NUM_VAL_IMAGES):
    labels = []
    names = []
    image_index = 0

    images = np.ndarray(shape=(batch_size, 64,64,3))
    val_images = os.listdir(testdir + '/images/')
    print(val_images)
    # Loop through all the images of a val directory
    batch_index = 0

    for image in val_images:
        image_file = os.path.join(testdir, 'images/', image)
        # print (testdir, image_file)

        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file)
        # print(image_index)
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data
            image_index += 1
            labels.append(label_dict[image])
            names.append(image)
            batch_index += 1

        if (batch_index >= batch_size):
            break

    print("Loaded Validation images ", image_index)
    return (images, np.asarray(labels), np.asarray(names))


def get_next_batch(data,label,batchsize=50):
    for cursor in range(0, len(data), batchsize):
        batch = []
        batch.append(data[cursor:cursor + batchsize])
        batch.append(label[cursor:cursor + batchsize])
        yield batch


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def get_train_label():
    label_dict = {}
    with open('./tiny-imagenet-200/wnids.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset = line[:-1]  # remove \n
            label_dict[synset] = i
    return label_dict

# In[6]:
print(IMAGE_DIRECTORY, TRAINING_IMAGES_DIR, VAL_IMAGES_DIR)
val_data = pd.read_csv(VAL_IMAGES_DIR + '/val_annotations.txt', sep='\t', header=None,
                       names=['File', 'Class', 'X', 'Y', 'H', 'W'])
label_d,train_label = label_dict(val_data),get_train_label()
val_images, val_labels, val_files = load_validation_images(VAL_IMAGES_DIR, val_data,label_d)
print(val_images.shape)
print(val_labels)
print(val_files)
print('Get Validation Data')

training_images, training_labels, training_files = load_training_images(TRAINING_IMAGES_DIR)

shuffle_index = np.random.permutation(len(training_labels))
training_images = training_images[shuffle_index]
training_labels = training_labels[shuffle_index]
training_files = training_files[shuffle_index]
training_labels_encoded = [train_label[i] for i in training_labels]
val_labels_encoded = [train_label[i] for i in val_labels]
print(val_labels_encoded[:10])

# In[7]:

height = IMAGE_SIZE
width = IMAGE_SIZE
channels = NUM_CHANNELS
n_inputs = height * width * channels
n_outputs = 200

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 64,64,3], name="X")
y = tf.placeholder(tf.int32, shape=[None], name="y")
z = tf.placeholder(tf.float32, name='z')
mode = tf.placeholder(tf.bool, name='mode')

def two_layer(X_reshaped,z,mode):
    # input shape [-1, 64, 64, 3]
    conv1 = tf.layers.conv2d(
        inputs=X_reshaped,
        filters=32,
        kernel_size=[5, 5],
        padding='SAME',
        activation=tf.nn.relu,
        name="conv1")

    # shape after conv1: [-1, 64, 64, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='SAME',
        activation=tf.nn.relu,
        name="conv2")

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print(pool2)
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1,16*16*64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=z, training=(mode == 'train'))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=200, name='output')

    return logits

print('Start Training')
learning_rate = 0.001
logits = two_layer(X,z,'train')
#logits = AlexNet(X_reshaped,z,200,None).fc8
print(logits,y)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9,use_nesterov=True).minimize(loss)

correct_train = tf.nn.in_top_k(logits,y,1)
accuracy = tf.reduce_mean(tf.cast(correct_train, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# In[9]:

n_epochs = 25
batch_size = 64
eval_size = 30
retrain = False
drop_out = 0.5
with tf.Session() as sess:
    if retrain == True:
        saver.restore(sess,'./models/tiny_imagenet_2layer_nodrop')
    else:
        init.run()
    epoch_train_loss,epoch_val_acc = [],[]
    for epoch in range(n_epochs):

        batch_index = 0
        for batch in get_next_batch(training_images,training_labels_encoded,batch_size):
            X_batch, y_batch = batch[0], batch[1]
            sess.run(optimizer, feed_dict={X: X_batch, y: y_batch, z:drop_out,mode:'train'})
            batch_index += 1
            # print('Batch accuracy is: ',sess.run(accuracy,feed_dict={X: X_batch, y: y_batch, z:drop_out,mode:'train'}))

        val_acc_list,val_loss_list = [],[]
        for batch in get_next_batch(val_images,val_labels_encoded,batch_size):
            X_batch, y_batch = batch[0], batch[1]
            acc_val = sess.run(accuracy,feed_dict={X: X_batch, y: y_batch, z: 1,mode:'val'})
            val_loss = sess.run(loss, feed_dict={X: X_batch, y: y_batch, z: 1, mode: 'val'})
            val_acc_list.append(acc_val)
            val_loss_list.append(val_loss)

        print(epoch, "Validation accuracy:", np.mean(np.array(val_acc_list), ' Loss :', ))

        # update accuracy
        epoch_val_acc.append(np.mean(np.array(val_acc_list)))
        epoch_train_loss.append(np.mean(np.array(val_loss_list)))

        save_path = saver.save(sess, "./models/tiny_imagenet_2layer_drop")
    x_axis = np.arange(n_epochs)
    plt.plot(x_axis,np.array(epoch_val_acc))
    plt.xlabel('Number of epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right',labels=['val accuracy'])
    plt.show()