import cv2
import time
import mediapipe as mp
from threading import Thread
class VidStream(object):
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.curtime = 0
        self.prevtime = 0
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.thread = Thread(target = self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status,self.frame) = self.capture.read()

    def show_frame(self):
        #(self.status, self.frame) = self.capture.read()
        if self.status:
            imgRGB = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLandmarks in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(self.frame, handLandmarks, self.mpHands.HAND_CONNECTIONS)
            self.curtime = time.time()
            fps = 1 / (self.curtime - self.prevtime)
            self.prevtime = self.curtime
            cv2.putText(self.frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 0, 0), 3)
            cv2.imshow("video",self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)


if __name__ == '__main__':
    vid_stream = VidStream(0)
    while True:
        try:
            vid_stream.show_frame()
        except AttributeError:
            pass