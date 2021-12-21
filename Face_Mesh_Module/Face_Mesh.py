import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpFace_mesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
Face_Mesh = mpFace_mesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness  = 1 , circle_radius= 1)
prevTime = 0
curTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Face_Mesh.process(imgRGB)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,face_landmarks,mpFace_mesh.FACEMESH_CONTOURS)
    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 255, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)