import cv2
import mediapipe as mp
import time
#This script is to be used as a module

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 #Previous Time
cTime = 0 #Current Time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) #Position of teh center
                print(id, cx, cy)
                if id == 4: #The id is one of the points n teh hand, tehere are 21 points, 0 is the center at the bone, 4 is the thumb tip
                    cv2.circle(img, (cx, cy), 15, (255, 255, 255), cv2.FILLED )
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime #The previous time becomes the current time

        #Putting the time on teh screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 255), 2) 
    cv2.imshow("Image", img)
    cv2.waitKey(1)