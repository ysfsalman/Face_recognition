# Code to collect facial dataset for face recognition training
# To feed the facial dataset to DL model, the program will acquire the dataset through following steps: 
#   1. Get the frame/image from webcam streaming
#   2. Gray the image
#   3. Detect face area by detection model from grayed image
#   4. Save and cropped face from image
#   5. (optional) show preview
#
# nb: some codes are collected from several sources
# ---------------------------------------------------------------------------------------------------------

import cv2

# get a classifier model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# make capture object
vc = cv2.VideoCapture(0)
 
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
i = 0
cv2.namedWindow("Record")
while rval:
    
    # get frame
    rval, frame = vc.read()

    # filter frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # classify face from grayed image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # acquire face image dataset and resize to a standard dimension
    for (x, y, w, h) in faces:
        print(x,y,w,h) # to check suitable standard dimension
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw rectangle on the frame
        
        # dataset resize
        dim = (125,125)
        graysized = cv2.resize(gray[y:y+h,x:x+w],dim,interpolation=cv2.INTER_AREA)

        # store frame
        name = 'Yusuf/'+str(i)+'.png'
        cv2.imwrite(name, graysized)
        i = i+1
    
    cv2.imshow('Record', frame)    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")

