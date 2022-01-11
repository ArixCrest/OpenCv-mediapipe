import cv2
import mediapipe as mp
import time

class FaceMesh():
    """static is for static images otherwise for videos it should be false
    if detailed is turned on it will track irises and more landmarks on lips
    Rest are self-explanatory"""
    def __init__(self,static=False, maxfaces=1,detailed = False,det_con = 0.5,track_con=0.5):
        self.statc = static
        self.maxfaces = maxfaces
        self.detailed = detailed
        self.det_con = det_con
        self.track_con = track_con

        self.mpFace_mesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.Face_Mesh = self.mpFace_mesh.FaceMesh(self.statc,self.maxfaces,self.detailed,self.det_con,self.track_con)
        "Modifying the color of connetion lines and landmarks"
        self.landmarkspec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1,color =(100,100,255))
        self.connectionspec = self.mpDraw.DrawingSpec(thickness=2,circle_radius=1,color=(0,255,0))
    """Return a list of list contaning the face number and the corresponding landmarks and their
        position in pixel value."""
    def findfaces(self,img,Draw = False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.Face_Mesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(img, face_landmarks, self.mpFace_mesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=self.landmarkspec,connection_drawing_spec=self.connectionspec)
                face = []
                h,w,c = img.shape
                for id, lnd in enumerate(face_landmarks.landmark):
                    x,y = int(lnd.x*w),int(lnd.y*h)
                    face.append([x,y])
                faces.append(face)

        return img,faces


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    curTime = 0
    detector = FaceMesh(detailed=True)
    while True:
        success, img = cap.read()
        img, faces = detector.findfaces(img,True)
        for i in faces:
            print(faces)
        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 255, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
