# import library necessary
import os

import cv2
import numpy as np
import pandas as pd
from keras import callbacks
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model
from keras.optimizer_v1 import Adagrad
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from numpy import concatenate
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import BatchNormalization


def initModelTraining(input__shape, num_labels):
    # Định nghĩa model
    model__fer = Sequential()

    # create a Sequential model incrementally
    # add convolution 2D
    # Thêm Convolutional layer với 64 kernel, kích thước kernel 3*3
    # dùng hàm relu làm activation và chỉ rõ input_shape cho layer đầu tiên
    model__fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input__shape))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model__fer.add(Dropout(0.5))

    # 2nd convolution layer, 2D convolution layer (spatial convolution over images)
    model__fer.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model__fer.add(Dropout(0.5))

    # 3rd convolution layer
    model__fer.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model__fer.add(BatchNormalization())
    model__fer.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten layer chuyển từ tensor sang vector
    model__fer.add(Flatten())

    # fully connected neural networks
    # Thêm Fully Connected layer với 1024 nodes và dùng hàm relu
    model__fer.add(Dense(1024, activation='relu'))  # Activate linear unit Rectified
    model__fer.add(Dropout(0.2))
    model__fer.add(Dense(1024, activation='relu'))
    model__fer.add(Dropout(0.2))

    # evaluate the categorical probabilities of the input data by softmax
    # Output layer với số num_labels node và dùng softmax function để chuyển sang xác suất.
    model__fer.add(Dense(num_labels, activation='softmax'))

    # Compliling the model
    # Compile model, chỉ rõ hàm loss_function nào được sử dụng, phương thức
    # đùng để tối ưu hàm loss function.
    model__fer.compile(loss=categorical_crossentropy,  # Computes the categorical crossentropy loss
                       optimizer=Adam(),  # Optimizer that implements the Adam algorithm
                       metrics=['accuracy'])  # judge the performance of your model
    return model__fer

# init size image and number of filters
num_features = 64
num_labels = 7
batch_size = 64
num_epochs = 1
width, height = 48, 48

# TRAINING CK+ DATASET
data_path = '../data/CKPlus/ck/CK+48'
data_dir_list = os.listdir(data_path)

img_data_list, img_data = [], []

for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        input_img_resize = cv2.resize(input_img, (48, 48))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data / 255
img_data.shape

num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

labels[0:134] = 0  # 135
labels[135:188] = 1  # 54
labels[189:365] = 2  # 177
labels[366:440] = 3  # 75
labels[441:647] = 4  # 207
labels[648:731] = 5  # 84
labels[732:980] = 6  # 249

names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)

# Split the dataset
X_train_ck, X_test_ck, y_train_ck, y_test_ck = train_test_split(x, y, test_size=0.15, random_state=2)
x_test_ck = X_test_ck

# init model
model_Ck = initModelTraining((48, 48, 3), num_labels)

# Get info model
model_Ck.summary()
model_Ck.get_config()
model_Ck.layers[0].get_config()
model_Ck.layers[0].input_shape
model_Ck.layers[0].output_shape
model_Ck.layers[0].get_weights()
np.shape(model_Ck.layers[0].get_weights()[0])
model_Ck.layers[0].trainable

# write log info training model
filename = 'model_train_new.csv'
filepath = "Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"
csv_log = callbacks.CSVLogger(filename, separator=',', append=False)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log, checkpoint]
callbacks_list = [csv_log]

model_Ck.fit(X_train_ck,
             y_train_ck,
             batch_size=7,
             epochs=num_epochs,
             verbose=1,
             validation_data=(X_test_ck, y_test_ck),
             callbacks=callbacks_list)
