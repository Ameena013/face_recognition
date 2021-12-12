import cv2
import numpy as np
import face_recognition

imgjim=face_recognition.load_image_file('faces/BarackObama1.jpg')
imgjim=cv2.cvtColor( imgjim,cv2.COLOR_BGR2RGB )

imgtest=face_recognition.load_image_file('faces/Test.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgjim)[0]
encodejim= face_recognition.face_encodings(imgjim)[0]
cv2.rectangle(imgjim,(faceloc[3],faceloc[0],faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodejimtest= face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0],faceloctest[1],faceloctest[2]),(255,0,255),2)

results= face_recognition.compare_faces([encodejim],encodejimtest)
faceDis = face_recognition.face_distance([encodejim],encodejimtest)
print(results,faceDis)
cv2.putText(imgtest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)

cv2.imshow('Jim1', imgjim)
cv2.imshow('Test', imgjim)
cv2.waitKey(0)

