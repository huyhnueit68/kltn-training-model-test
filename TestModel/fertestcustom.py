# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np
import cv2

# loading the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x = None
y = None
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# loading image
cap = cv2.VideoCapture(0)  # bật webcam
full_size_image = cap
while True:
    ret, test_img = full_size_image.read()  # captures frame and returns boolean value and captured image
    # test_img[:, :, 0] = cv2.equalizeHist(test_img[:, :, 0])
    # test_img = cv2.GaussianBlur(test_img, (5, 5), cv2.BORDER_DEFAULT)

    if not ret:
        continue
    # histogram

    print("Image Loaded")
    # Convert our original image from the BGR color space to gray
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray_img, 1.3, 10)

    # detecting faces
    for (x, y, w, h) in faces:
        roi_gray = gray_img[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.imshow('Anh truoc khi nhan dien', test_img)
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # predicting the emotion
        yhat = loaded_model.predict(cropped_img)
        cv2.putText(test_img, labels[int(np.argmax(yhat))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                    1, cv2.LINE_AA)
        print("Cam xuc: " + labels[int(np.argmax(yhat))])
        if labels[int(np.argmax(yhat))] == 'Disgust':
            break

    cv2.imshow('Anh sau khi nhan dien', test_img)
    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break
