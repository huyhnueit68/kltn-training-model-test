# import libraly necessary librari to run program include:
# 1. pip install opencv-python
# 2. pip install keras

import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import time

# Test model FER 2013 phiên bản dùng được cả realtime lẫn ảnh
# Có tích hợp sẵn cân bằng histogram và lọc gauss
# CreatedBy: PQ Huy

def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    print(cumulator)
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

# load model
model = model_from_json(open("../model/fer2.json", "r").read())
# model = model_from_json(open("ckplus02022022.json", "r").read())

# load weights
# model.load_weights('ckplus02022022.h5')
model.load_weights('../model/fer2.h5')

# detec face
# cắt nhỏ từng ảnh ko nhận dạng được -> xem face_haar_cascade nhận dạng tốt hơn ko
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # training lại model này cho nhận dạng mặt tốt hơn

cap = cv2.VideoCapture(0)  # bật webcam
# cap = cv2.VideoCapture("nckh002.png", cv2.CAP_IMAGES)

result = []
angry = 0
disgust = 0
fear = 0
happy = 0
sad = 0
surprise = 0
neutral = 0
yLable = [angry, disgust, fear, happy, sad, surprise, neutral]

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    # test_img[:, :, 0] = cv2.equalizeHist(test_img[:, :, 0])
    # test_img = cv2.GaussianBlur(test_img, (5, 5), cv2.BORDER_DEFAULT)

    if not ret:
        continue
    # Convert our original image from the BGR color space to gray
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # histogram

    # mảng gồm chiều cao, rộng, tọa độ x, tọa độ y
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    # define emotions
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    xLable = ['Tức giận', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn', 'Ngạc nhiên', 'Bình thường']

    timer = 0
    count_face = 0
    start_time = time.time()
    for (x, y, w, h) in faces_detected:

        # vẽ 1 hình cn dựa trên x, y, w, h
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

        # cropping region of interest i.e. face area from  image
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(gray_img, (48, 48)) # resize giống kích thước training model

        # conver a PIL Image instance to a Numpy array.

        # covert lại dữ liệu
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # if count_face != 0:
        #     timer += time.time() - start_time
        # count_face += 1

        # find max indexed array, returns the indices of the maximum values along an axis.
        max_index = np.argmax(predictions[0])

        # calc total emotion
        # ảnh 1 ( 0.1% là giận, 5 % buồn )
        angry += predictions[0][0]
        disgust += predictions[0][1]
        fear += predictions[0][2]
        happy += predictions[0][3]
        sad += predictions[0][4]
        surprise += predictions[0][5]
        neutral += predictions[0][6]

        predicted_emotion = emotions[max_index]

        # set emotional state
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        yLable = [angry, disgust, fear, happy, sad, surprise, neutral]

    # hiển thị thời gian đánh giá
    print("--- tổng time " + str(time.time() - start_time))

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Facial emotion analysis ", resized_img)
    # plotting
    plt.title("")
    plt.xlabel("Loại cảm xúc")
    plt.ylabel("Tỷ lệ cảm xúc")
    plt.bar(xLable, yLable, width=0.2)
    plt.show()

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

count_face = count_face - 1
print("Trung bình" + str(timer/count_face))
print("Tổng thời gian: " + str(timer))

cap.release()
cv2.destroyAllWindows
