import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils
prevTime = 0
curTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            bbox = int(bboxC.xmin*w) , int(bboxC.ymin*h), int(bboxC.width*w),int(bboxC.height*h)
            cv2.rectangle(img,bbox,(0,255,0),2)
            cv2.putText(img,f'{int((detection.score[0]*100))}%',
                        (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 255, 0),2)

    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 255, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)