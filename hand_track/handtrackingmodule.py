import cv2
import mediapipe as mp
import time
class handdetector():
    def __init__(self,mode=False,max_hands=5,model_complex=1,detection_con=0.5,track_con=0.5):
        self.mode=mode
        self.max_hands=max_hands
        self.model_complex=model_complex
        self.detection_con=detection_con
        self.track_con=track_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,self.model_complex,self.detection_con,self.track_con)
        self.mpDraw = mp.solutions.drawing_utils
    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findposition(self,img, handNo=0 ,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                #     # print(id,lm)
                height,weight,channel=img.shape
                channelx ,channely=int(lm.x*weight),int(lm.y*height)
                    # print(id,channelx,channely)
                lmlist.append([id,channelx,channely])
                if draw:

                #     # if id==4:
                #     # remove this condition and it will draw on all of them
                    cv2.circle(img,(channelx,channely),15,(0,0,255),cv2.FILLED)
        return lmlist
def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector=handdetector()

    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist=detector.findposition(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()