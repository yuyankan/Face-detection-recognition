"""
Real Time Face recognition:
-> Each face stored on dataset/dir, should have a unique numeric integer ID as 1, 2, 3
-> LBPH computed model (trained faces) should be on trainer/dir
cBased on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
"""
import cv2
import numpy as np
import os

# upload trained face recognizer 'trainer_f.yml' & face detector: ''Cascades/haarcascade_frontalface_default.xml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer_f.yml')
cascadepath = 'Cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadepath)

font = cv2.FONT_HERSHEY_SIMPLEX

# iniciate id counter
id = 0

# names related to ids:
names = ['guo er', 'me', 'wei']

# Test set: sample list
path = 'test'
sample_path = []
images = []
for sub_path in os.listdir(path):
    temp_path = os.path.join(path, sub_path)
    sample_path.append(temp_path)

    image_temp = cv2.resize(cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE), (450, 600))
    images.append(image_temp)

minW = int(0.1* images[0].shape[0])
minH = int(0.1* images[0].shape[1])

for image_2 in images:

    faces = faceCascade.detectMultiScale(image_2,
                                         scaleFactor=1.1,
                                         minNeighbors=5,

                                         )
    print(faces)
    for (x, y, w, h) in faces:
        # make box mark on pic.
        cv2.rectangle(image_2, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(image_2[x: x+w, y: y+h])

        if confidence < 100:
            id = names[id]
        else:
            id = 'unknow'
            confidence = '  {0}%'.format(round(100-confidence))
        # put Text on pic.
        cv2.putText(image_2, str(id), (x+5, y-5),font, 1, (255, 255, 255), 2)
        cv2.putText(image_2, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 255), 2)

    cv2.imshow('image_2', image_2)

    #'ESC' -->exit
    k = cv2.waitKey(5000)&0xff
    if k ==27:
        break

print('\n [INFO] Existing program and cleanup stuff')
cv2.destroyAllWindows()