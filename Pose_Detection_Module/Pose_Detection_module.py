import cv2
import mediapipe as mp
import time


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
class PoseDetection():
    def __init__(self,mode = False,model_complex = 1,smooth_lds = True,segmentation = False,
                 smooth_seg = True,det_conf = 0.5,track_conf = 0.5):
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(mode,model_complex,smooth_lds,
                                     segmentation,smooth_seg,det_conf,track_conf)
    def find_pose(self, img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    """ will return the positions of landmarks
        if no id_no is given it will return a list of list of all the positions
        otherwise if id_no is given 0-20 then only that id will be returned"""
    def find_position(self,img,id_no = -1,draw = False):
        pos_list = []
        h, w, c = img.shape
        if self.results.pose_landmarks:
            my_pose = self.results.pose_landmarks
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if (id_no == -1):
                    pos_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy),
                                   10, (0, 0, 0), cv2.FILLED)
                else:
                    if (id == id_no):
                        pos_list.append([cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy),
                                       10, (0, 0, 0), cv2.FILLED)
        return pos_list


def main():
    cap = cv2.VideoCapture("videos/2.mp4")
    prevTime = 0
    curTime = 0
    detector = PoseDetection()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        poslist = detector.find_position(img,15,True)
        if len(poslist)!=0:
            print(poslist)
        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()