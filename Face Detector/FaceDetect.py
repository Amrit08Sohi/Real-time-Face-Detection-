import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
ptime = 0
mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face = mpFace.FaceDetection(0.7)
while True:
    success, img = cap.read()
    img = cv2.resize(img,(640,480))
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    # print(results)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(detection.score)
            # print(id,detection)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            # mpDraw.draw_detection(img,detection)
            # to get the pixel values of bounding box
            bbox = int(bboxC.xmin * w),int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h) 
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'{int(detection.score[0] * 100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)


    ctime = time.time()
    fps = 1 / (ctime-ptime)
    ptime = ctime

    cv2.putText(img,f'FPS : {int(fps)}',(40,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) == 27:
        break