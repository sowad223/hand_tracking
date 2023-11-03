import cv2
import mediapipe as mp
import time
import handtrackingmodule as sowad
previous_time = 0
current_time = 0
cap = cv2.VideoCapture(0)
detector=sowad.handdetector()
while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist=detector.findposition(img,draw=False)
        if len(lmlist)!=0:
            print(lmlist[4])
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
