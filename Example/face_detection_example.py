import numpy as np

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
filename = 'face.tiff'
img = cv2.imread('face.tiff')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray,(96,96))
    cv2.imwrite("process_"+filename,roi_gray)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()