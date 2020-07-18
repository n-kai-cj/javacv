import cv2
import numpy as np

width = 640
height = 480
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.open(0)
prevgray_cuda = None

optflow_cuda = cv2.cuda.FarnebackOpticalFlow_create(numLevels=5, pyrScale=0.5, fastPyramids=False,
    winSize=13, numIters=10, polyN=5, polySigma=1.1, flags=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prevgray_cuda is None:
        prevgray_cuda = cv2.cuda_GpuMat(gray.copy())
        continue

    # calculate dense optical flow
    gray_cuda = cv2.cuda_GpuMat()
    gray_cuda.upload(gray)
    flow_cuda = optflow_cuda.calc(prevgray_cuda, gray_cuda, None)
    flow = flow_cuda.download()

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

    prevgray_cuda = gray_cuda


cap.release()
cv2.destroyAllWindows()
