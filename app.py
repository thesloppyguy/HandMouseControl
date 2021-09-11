
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import HTmodule
# import test
import autopy


wCam, hCam = 640, 480
smoothening = 5
wScr, hScr = autopy.screen.size()

plocX, plocY = 0, 0
clocX, clocY = 0, 0

t1 = 0
t2 = 0
cam = cv.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
module = HTmodule.Hand_Detector(detection_conf=0.75, track_conf=0.75)

# module = test.handDetector()

while True:
    success, frame = cam.read()
    cv.rectangle(frame, (100, 100),
                 (wCam-100, hCam-100), (255, 255, 0), 2)
    # get hand
    frame = module.find_hands(frame)
    # get finger tips
    lmlist = module.find_position(frame)
    if len(lmlist) != 0:
        print(lmlist[8], lmlist[12])
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[8][2]
        # which fingers are up
        fing = module.finger_up()
        print(fing)

        # one finger
        if fing[1] == 1 and fing[2] == 0:
            x3 = np.interp(x1, (0, wCam-100), (0, wScr))
            y3 = np.interp(y1, (0, hCam-100), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(clocX, clocY)
        # two fingers
        if fing[1] == 1 and fing[2] == 1 and fing[3] == 1:
            length, frame, info = module.find_dist(8, 12, frame)
            if length < 40:
                cv.circle(frame, (info[4], info[5]),
                          15, (255, 255, 0), cv.FILLED)
                autopy.mouse.click(delay=10)

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
