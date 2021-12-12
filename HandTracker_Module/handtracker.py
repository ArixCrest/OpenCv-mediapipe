import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
prevTime = 0
curTime = 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH,800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,800)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w , c = img.shape
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                #Tracks 8th landmark point and draws a circle there
                if(id==8):
                    cx , cy = int(lm.x*w), int(lm.y*h)
                    print(id, cx, cy)
                    cv2.circle(img, (cx,cy),15, (255,0,0), cv2.FILLED)
            mpDraw.draw_landmarks(img,handLandmarks,mpHands.HAND_CONNECTIONS)

    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255,0,0), 3)
    cv2.imshow("image",img)
    cv2.waitKey(1)
