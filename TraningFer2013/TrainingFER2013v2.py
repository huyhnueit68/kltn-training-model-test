import numpy as np
import pandas as pd
import cv2
import glob as gb
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# Traning model với FER2013 phiên bản đọc dữ liệu thô và xử lý (thử nghiệm)
# CreatedBy: PQ Huy

def getcode(n):
    for x, y in code.items():
        if n == y:
            return x


s = 48
code = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

X_train = []
y_train = []
train_dir = '../data/Fer2013/train/'

X_test = []
y_test = []
test_dir = '../data/Fer2013/test/'

# train
for folder in os.listdir(train_dir):
    files = gb.glob(pathname=str(train_dir + folder + '/*.jpg'))
    for file in files:
        image = cv2.imread(file)
        X_train.append(list(cv2.resize(image, (s, s))))
        y_train.append(code[folder])

# test
for folder in os.listdir(test_dir):
    files = gb.glob(pathname=str(test_dir + folder + '/*.jpg'))
    for file in files:
        image = cv2.imread(file)
        X_test.append(list(cv2.resize(image, (s, s))))
        y_test.append(code[folder])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

X_train = X_train / 255.0
X_test = X_test / 255.0
plt.figure(figsize=(20, 20))

for n, i in enumerate(list(np.random.randint(0, len(X_train), 36))):
    plt.subplot(6, 6, n + 1)
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title(getcode(y_train[i]))

for n, i in enumerate(list(np.random.randint(0, len(X_test), 36))):
    plt.subplot(6, 6, n + 1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(getcode(y_train[i]))

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("-------------------------------")

print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
print("-------------------------------")

model = Sequential([
    Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', input_shape=(48, 48, 3)),
    MaxPooling2D(3),
    Conv2D(128, 3, activation='relu', kernel_initializer='he_normal'),
    Conv2D(256, 3, activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D(3),
    Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal'),
    MaxPooling2D(3),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(256, activation='relu'),
    Dense(7, activation='softmax', kernel_initializer='glorot_normal')
])

es = EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=2,
    mode='max', baseline=None, restore_best_weights=True
)
lr = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=5, verbose=2,
    mode='max', min_delta=1e-5, cooldown=0, min_lr=0
)

callbacks = [es, lr]
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=8, epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[callbacks],
                    shuffle=True)

model.evaluate(X_train, y_train)
model.evaluate(X_test, y_test)

# Saving the  model
fer_json = model.to_json()
with open("fer03022022.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer03022022.h5")
