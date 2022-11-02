import cv2
import numpy
import face_recognition

#step1
#converting BGR to RGB as this is what understood by computer

# for Traning
imgElon = face_recognition.load_image_file('Imagebasic/Elon Musk.jpg')#Importing Image
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)# convertion
#for Testing
imgTest = face_recognition.load_image_file('Imagebasic/Elon Test.jpg')#Importing Image
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)# convertion
#step2 Finding Faces and encodings of them
# face location
faceLoc= face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
#for knowing Face Locations
#print(faceLoc)
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#for testing
faceLocTest= face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#step3 Comparing These Faces and Fingding The Distance Between Them(128 measurments of both faces ).
#Linear SVM whether they Match or not
results = face_recognition.compare_faces([encodeElon],encodeTest)
# To know best Match we find Distance(The lower the Better)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
# To display on Image
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)
# Nwxt is an Attendence project based on this library