import cv2 as cv
import mediapipe as mp
import time
import HTmodule
import autopy


t1 = 0
t2 = 0
cam = cv.VideoCapture(0)
detector = Hand_Detector()

while True:
    success, frame = cam.read()
    frame = detector.find_hands(frame)
    lmList = detector.find_position(frame)

    # fps counter
    t1 = time.time()
    fps = 1/(t1-t2)
    t2 = t1
    cv.putText(frame, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 255), 1)
    cv.imshow("video", frame)
    # fps counter ends

    # exit
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
