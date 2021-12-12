import cv2
import numpy as np
import face_recognition
import os


path = 'atten'
images=[]
classesName = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curIMG =cv2.imread(f'{path}/{cl}')
    images.append(curIMG)
    classesName.append(os.path.splitext(cl[0]))
print(classesName)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknow = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success , img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facescurfame = face_recognition.face_locations(imgS)
    encodecurfame = face_recognition.face_encodings(imgS,facescurfame)

    for encodefaces , faceLoc in zip(encodecurfame,facescurfame):
        matches = face_recognition.compare_faces(encodeListknow,encodefaces)
        facdis = face_recognition.face_distance(encodeListknow ,encodefaces)
        #print(faceDis)
        matchindex = np.argmin(facdis)

        if matches[matchindex]:
            name = classesName[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 +y1*4,x2*4,y2,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

cv2.imshow('webcam',img)
cv2.waitkey(1)