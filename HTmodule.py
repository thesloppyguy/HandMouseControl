import cv2 as cv
import mediapipe as mp
import time


class Hand_Detector():

    def __init__(self, mode=False, Max_hands=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.Max_hands = Max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
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
        lmList = []
        if self.results.multi_hand_landmarks:
            hand_points = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(hand_points.landmark):
                h, w, c = frame.shape
                y, x = int(h*lm.y), int(w*lm.x)
                lmList.append([id, x, y])
                if draw:
                    cv.circle(frame, (x, y), 10, (255, 0, 255), cv.FILLED)
        return lmList


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
