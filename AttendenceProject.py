import cv2
import numpy
import face_recognition
import os
# step1 same but now create a list that will imoprt all images automatically from folder.
import numpy as np
from datetime import datetime
path = 'ImageAttendence'
images = []
classnames = []
# grab the images of this folder
myList=os.listdir(path)
print(myList)
# using these names and import one by one
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0]) # we dont need .jpg here we only need names
print(classnames)
# functon for encoding each image
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convertion
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
# marking attendence in CSV file.
def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)
        # for entering the date time and name of person recognised
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0]) # name
        # checking if current name is there or not
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')





# calling encoding function for known faces
encodeListknown = findEncodings(images)
print(len(encodeListknown)) # to know how many encodings
print('Encoding Complete')
# for for testing we get Image from webcam
# To initialize webcam
cap = cv2.VideoCapture(0)
# to get each frame one by one
while True:
    success , img = cap.read()
    # to reduce size of image
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS) # there will me mulitple images to find location of our faces
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)
    # Final step Iterate through all faces and compare all with encodings we found before
    # 1 by 1 it wil grab one face location and encode face from FaceCurFrame,encodeCurFrame in same loop.
    # so we used zip
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)#gives the distance of each one of them
        print(faceDis) # to find Best Match
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name= classnames[matchIndex].upper()
            print(name)
            # to show face loacation and also we need to scale it x4 time since we compressed before
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x2,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

    # show the image as well

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




# faceLoc= face_recognition.face_locations(imgElon)[0]
# encodeElon=face_recognition.face_encodings(imgElon)[0]
# #for knowing Face Locations
# #print(faceLoc)
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# #for testing
# faceLocTest= face_recognition.face_locations(imgTest)[0]
# encodeTest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# #step3 Comparing These Faces and Fingding The Distance Between Them(128 measurments of both faces ).
# #Linear SVM whether they Match or not
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# # To know best Match we find Distance(The lower the Better)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)