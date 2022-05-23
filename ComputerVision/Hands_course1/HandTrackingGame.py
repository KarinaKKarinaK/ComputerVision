import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

pTime = 0 #Previous Time
cTime = 0 #Current Time
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4]) #now You can print teh position at any index, teh index being teh number assosciated with teh point

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime #The previous time becomes the current time

    #Putting the time on teh screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 255), 2) 
    cv2.imshow("Image", img)
    cv2.waitKey(1)