import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self,min_detectionCon = 0.5):
        self.min_detectionCon = min_detectionCon
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face = self.mpFace.FaceDetection(self.min_detectionCon)


    def findFaces(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        # print(self.results)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(detection.score)
                # print(id,detection)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                h,w,c = img.shape
                # mpDraw.draw_detection(img,detection)
                # to get the pixel values of bounding box
                bbox = int(bboxC.xmin * w),int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h) 
                bboxes.append([id,bbox,detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img,f'{int(detection.score[0] * 100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        return bboxes,img
    

    def fancyDraw(self,img,bbox,l = 30, thickness = 6,rt = 1):
        x,y,w,h = bbox
        # bottom right point
        x1,y1 = x+w,y+h

        cv2.rectangle(img,bbox,(255,0,255),rt)
        # Top left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),thickness)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),thickness)
        # Top right x1,y1
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),thickness)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),thickness)
        # Bottom left
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),thickness)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),thickness)
        # Bottom Right
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),thickness)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),thickness)
        return img

def main(): 
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(640,480))

        bboxes,img = detector.findFaces(img)
        # print(bboxes)
        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime

        cv2.putText(img,f'FPS : {int(fps)}',(40,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

        cv2.imshow("Image1",img)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    main()