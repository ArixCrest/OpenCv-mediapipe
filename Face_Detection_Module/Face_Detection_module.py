import cv2
import mediapipe as mp
import time

class Facedetector():
    """Minconfidence(recommended is .75 or higher .5 sometimes detects dog faces
        if for confidence detection And range is for distance 0 is better for close
        faces like 2m for faces that are far like 5m then use range = 1."""
    def __init__(self,minconfindence=0.75,range=0):
        self.confidence = minconfindence
        self.range = range
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.confidence,self.range)
        self.mpDraw = mp.solutions.drawing_utils
    "draw is false by default"
    def get_bounding_box(self,img,draw = False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face.process(imgRGB)
        boxes = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                boxes.append([bbox,detection.score[0]])
                if(draw):
                    img = self.draw_boxes(img,bbox,detection.score[0])

        return img,boxes
    "Draws the boxes"
    def draw_boxes(self,img,bbox,confidence):
        cv2.rectangle(img, bbox, (0, 255, 0), 2)
        cv2.putText(img, f'{int((confidence * 100))}%',
                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 0), 2)
        return img
def main():
    prevTime = 0
    curTime = 0
    cap = cv2.VideoCapture(0)
    detector = Facedetector(0.75)
    while True:
        success, img = cap.read()
        img, boxes = detector.get_bounding_box(img,True)
        for i in range(len(boxes)):
            print(i, boxes[i])
        curTime = time.time()
        fps = 1 / (curTime - prevTime)
        prevTime = curTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 255, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ =="__main__":
    main()