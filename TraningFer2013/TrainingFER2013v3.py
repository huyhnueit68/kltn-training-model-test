# import library necessary

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
from keras.regularizers import l2


# Traning model với FER2013 phiên bản đọc dữ liệu từ file CSV có sẵn mà không cần xử lý dữ liệu thô
# CreatedBy: PQ Huy

# Hàm khởi tạo model
def initModelTraining(input__shape):
    # call model Sequential
    model = Sequential()

    # desinging the CNN
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1),
                     data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.2))

    # evaluate the categorical probabilities of the input data by softmax
    model.add(Dense(num_labels, activation='softmax'))
    model.summary()
    # Compliling the model
    model.compile(loss=categorical_crossentropy,  # Computes the categorical crossentropy loss
                  optimizer='adam',  # Optimizer that implements the Adam algorithm
                  metrics=['accuracy'])  # judge the performance of your model

    return model


# def get_transforms_input_data(_data):
#     if 'Training' in _data['Usage']:
#         return A.Compose(
#             [
#                 # thay đổi độ sáng và độ tương phản của ảnh brightness_limit: thay đổi độ sáng giới hạn trong phạm vi (-0.2,0.2)
#                 # contrast_limit: thay đổi độ tương phản trong phạm vi (-0.2,0.2)
#                 # p xác xuất biến đổi là 0.9
#                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
#                 A.Rotate(limit=30, p=0.9),
#                 # A.RandomResizedCrop((48,48), 48, 48),
#                 A.GaussNoise(),
#                 # Chuẩn hóa được áp dụng theo công thức:img = (img - mean * max_pixel_value) / (std * max_pixel_value)
#                 A.Normalize(
#                     mean=0.131,
#                     std=0.308
#                 ),
#                 ToTensorV2()
#             ]
#         )
#     elif 'PublicTest' in _data['Usage']:
#         return A.Compose(
#             [
#                 A.Normalize(
#                     mean=0.131,
#                     std=0.308
#                 ),
#                 ToTensorV2()
#             ]
#         )


# init size image and number of filters
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

# TRAINING DATASET OF FER2013
# connect and read file csv
df = pd.read_csv('../data/Fer2013/fer2013.csv')

# init 4 array for training
X_train, train_y, X_test, test_y = [], [], [], []
count = 0

# load all info in df and append to array
for index, row in df.iterrows():
    # Split a string
    val = row['pixels'].split(" ")
    try:
        # add element to the end of array with data format as float
        # if 'Training' in row['Usage'] or 'PrivateTest' in row['Usage']:
        if 'Training' in row['Usage'] or 'PrivateTest' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

X_train = np.array(X_train, 'float32')
train_y = np.array(train_y, 'float32')
X_test = np.array(X_test, 'float32')
test_y = np.array(test_y, 'float32')

# convert array to vertor, using num_classes set total class
train_y = np_utils.to_categorical(train_y, num_classes=num_labels)
test_y = np_utils.to_categorical(test_y, num_classes=num_labels)

# Compute the arithmetic mean along the specified axis and Compute the standard deviation along the specified axis.
X_train -= np.mean(X_train, axis=0)  # normalize data between 0 and 1
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

# matrix transpose
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# call model Sequential
model__fer = initModelTraining(X_train.shape[1:])

# load model
# model__fer = model_from_json(open("ferTest.json", "r").read())

# load weights
# model__fer.load_weights('ferTest.h5')
# Compliling the model
# model__fer.compile(loss=categorical_crossentropy,  # Computes the categorical crossentropy loss
#                    optimizer=Adam(),  # Optimizer that implements the Adam algorithm
#                    metrics=['accuracy'])  # judge the performance of your model

ResultModel = model__fer.fit(X_train, train_y,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1,
                             validation_data=(X_test, test_y),
                             shuffle=True)

# Saving the  model
fer_json = model__fer.to_json()
with open("fer03022022.json", "w") as json_file:
    json_file.write(fer_json)
model__fer.save_weights("fer03022022.h5")
