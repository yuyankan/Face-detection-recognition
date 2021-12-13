import cv2
import numpy as np
import os

# training set: 1. training path
images_path = []
ids = []
path = 'dataset/faces_pic'
for f in os.listdir(path):
    sub_path = os.path.join(path, str(f))
    id = int(f)
    for image_path in os.listdir(sub_path):
        images_path.append(os.path.join(path, str(f), image_path))
        ids.append(id)
print('ids:', ids)

# training set: 2. training samples list
images = []
for image in images_path:
    images.append(cv2.imread(image,cv2.IMREAD_GRAYSCALE))

# train : face recognizer
labels = np.array(ids)
recognizer = cv2.face.LBPHFaceRecognizer_create() # face recognizer model
cv2.face_LBPHFaceRecognizer.train(recognizer, images, labels) # train model
cv2.face_LBPHFaceRecognizer.write(recognizer, 'trainer/trainer_f.yml')

# dev/test set: face recognizer prediction
predict_image = []
predict_image.append(cv2.resize(cv2.imread('test/test_1.jpg', cv2.IMREAD_GRAYSCALE), (450, 600)))
predict_image.append(cv2.resize(cv2.imread('test/wei_3.jpg', cv2.IMREAD_GRAYSCALE), (450, 600)))

label_predict = []
confidence_predict = []

for image in predict_image:
    temp_label, temp_confidence = recognizer.predict(image)
    label_predict.append(temp_label)
    confidence_predict.append(temp_confidence)

print('label_predict: ', label_predict)
print('confidence_predict: ', confidence_predict)