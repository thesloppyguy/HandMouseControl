import cv2 as cv
import mediapipe as mp
import time
import math


class Hand_Detector():

    def __init__(self, mode=False, Max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.Max_hands = Max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.tipIds = [4, 8, 12, 16, 20]
        self.cam = cv.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.Max_hands, self.detection_conf, self.track_conf)

    def find_hands(self, frame, draw=True):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        if self.results.multi_hand_landmarks:
            for item in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame, item, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame, handno=0, draw=False):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            hand_points = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(hand_points.landmark):
                h, w, c = frame.shape
                y, x = int(h*lm.y), int(w*lm.x)
                self.lmList.append([id, x, y])
                if draw:
                    cv.circle(frame, (x, y), 10, (255, 0, 255), cv.FILLED)
        return self.lmList

    def finger_up(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            # totalFingers = fingers.count(1)
        return fingers

    def find_dist(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
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


if __name__ == '__main__':
    main()
