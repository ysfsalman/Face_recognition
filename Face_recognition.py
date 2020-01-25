import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
i = 0
 
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    
    rval, frame = vc.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        name = 'Yusuf/'+str(i)+'.png'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # dataset resize
        dim = (125,125)
        graysized = cv2.resize(gray[y:y+h,x:x+w],dim,interpolation=cv2.INTER_AREA)
        cv2.imwrite(name, graysized)
        i = i+1
    cv2.imshow('FaceDetection', frame)    
    
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")