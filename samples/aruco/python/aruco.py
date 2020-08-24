import os
import cv2
import numpy as np

# marker dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
dir_name = 'armarker'

# make markers
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
for i in range(5):
    file_name = '{}/{:02d}.png'.format(dir_name, i)
    if not os.path.exists(file_name):
        markerImage = cv2.aruco.drawMarker(dictionary, id=i, sidePixels=100)
        cv2.imwrite(file_name, markerImage)

# Camera parameter
fx = 540.627
fy = 550.577
cx = 320.833
cy = 240.796
k1 = 0.1546
k2 = -0.331
p1 = -0.00123
p2 = -0.0001

K = np.identity(3)
K[0, 0] = fx
K[1, 1] = fy
K[0, 2] = cx
K[1, 2] = cy
distCoeffs = (k1, k2, p1, p2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # show captured frame
    cv2.imshow('capture', frame)
    if cv2.waitKey(1) == 27:
        break
    
    # detect markers
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, K, distCoeffs)
    if markerIds is None:
        continue
    if len(markerIds) <= 0:
        continue

    # draw markers
    drawFrame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    # pose estimation
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.2, K, distCoeffs)
    for i in range(0, len(rvecs)):
        rvec = rvecs[i]
        tvec = tvecs[i]
        drawFrame = cv2.drawFrameAxes(drawFrame, K, distCoeffs, rvec, tvec, 0.2, 3)
    
    # show drawed frame
    cv2.imshow('marker', drawFrame)

cap.release()
cv2.destroyAllWindows()

