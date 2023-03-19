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
            landmarks = detection.location_data.relative_keypoints
            # print(landmarks)
            h,w,c = img.shape

            # left and right eye landmarks
            righteye = int(landmarks[0].x * w),int(landmarks[0].y * h)
            cv2.circle(img,righteye,5,(255,0,0),cv2.FILLED)
            lefteye = int(landmarks[1].x * w),int(landmarks[1].y * h)
            cv2.circle(img,lefteye,5,(255,0,0),cv2.FILLED)
            cv2.line(img,righteye,lefteye,(0,255,0),1)

            #  NodeTip landmarks
            noseTip = int(landmarks[2].x * w),int(landmarks[2].y * h)
            cv2.circle(img,noseTip,5,(255,0,0),cv2.FILLED)
            cv2.line(img,righteye,noseTip,(0,255,0),1)
            cv2.line(img,lefteye,noseTip,(0,255,0),1)


            #  Mouth landmarks
            mouth = int(landmarks[3].x * w),int(landmarks[3].y * h)
            cv2.circle(img,mouth,5,(255,0,0),cv2.FILLED)
            cv2.line(img,mouth,noseTip,(0,255,0),1)
            cv2.line(img,mouth,lefteye,(0,255,0),1)
            cv2.line(img,mouth,righteye,(0,255,0),1)

            #  RightEar landmarks
            rightEar = int(landmarks[4].x * w),int(landmarks[4].y * h)
            cv2.circle(img,rightEar,5,(255,0,0),cv2.FILLED)
            cv2.line(img,righteye,rightEar,(0,255,0),1)
            cv2.line(img,noseTip,rightEar,(0,255,0),1)
            cv2.line(img,mouth,rightEar,(0,255,0),1)
            
            # LeftEar Landmarks
            leftEar = int(landmarks[5].x * w),int(landmarks[5].y * h)
            cv2.circle(img,leftEar,5,(255,0,0),cv2.FILLED)
            cv2.line(img,lefteye,leftEar,(0,255,0),1)
            cv2.line(img,noseTip,leftEar,(0,255,0),1)
            cv2.line(img,mouth,leftEar,(0,255,0),1)
            # mpDraw.draw_detection(img,detection)
            # to get the pixel values of bounding box
            bbox = int(bboxC.xmin * w),int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h) 
            cv2.rectangle(img,bbox,(0,255,0),2)
            cv2.putText(img,f'{int(detection.score[0] * 100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)


    ctime = time.time()
    fps = 1 / (ctime-ptime)
    ptime = ctime

    cv2.putText(img,f'FPS : {int(fps)}',(40,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) == 27:
        break