import cv2
import numpy as np

width = 640
height = 480
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.open(0)
prevgray = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prevgray is None:
        prevgray = gray.copy()
        continue

    # calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.1,
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    # draw direction of flow
    for y in range(0, frame.shape[0], 20):
        for x in range(0, frame.shape[1], 20):
            flowatxy = flow[y, x]
            cv2.line(frame, (x, y), (int(
                x+flowatxy[0]), int(y+flowatxy[1])), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x, y), (int(
                x+flowatxy[0]), int(y+flowatxy[1])), (192, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 1, (0, 133, 255), 1, cv2.LINE_AA)

    cv2.imshow("Dense Optical Flow", frame)

    key = cv2.waitKey(10)
    if key == 27:
        break

    prevgray = gray.copy()


cap.release()
cv2.destroyAllWindows()
