import cv2
import mediapipe as mp
import time
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
previous_time=0
current_time=0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)


    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                height,weight,channel=img.shape
                channelx ,channely=int(lm.x*weight),int(lm.y*height)
                print(id,channelx,channely)
                if id==4:
                # remove this condition and it will draw on all of them
                    cv2.circle(img,(channelx,channely),15,(0,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img , handLms, mpHands.HAND_CONNECTIONS)
    current_time=time.time()
    fps=1/(current_time-previous_time)
    previous_time= current_time
    cv2.putText(img, str(int(fps)),(10,60),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,255),2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

