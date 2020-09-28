import os
import time
import cv2
import numpy as np
from pyquaternion import Quaternion
import open3d

vis = open3d.visualization.Visualizer()
vis.create_window(
    window_name='ArUco2',
    width=640,
    height=480,
    left=100,
    top=100
)

def show(name, frame):
    cv2.imshow(name, frame)
    key = cv2.waitKey(1)
    if key == 27:
        return False
    return True


def update_camera(K, R, t, scale=0.3):
    # intrinsics
    Kinv = np.linalg.inv(K/scale)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5*scale)
    axis.transform(T)
    vis.add_geometry(axis)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    color = [0.9, 0.7, 0.7]
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)
    vis.add_geometry(plane)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [1, 2],
        [2, 4],
        [4, 3],
        [3, 1],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    color = [0.8, 0.2, 0.2]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)


def update_trajectory(last_t, t):
    color=[0.8, 0.2, 0.2]
    colors = [color for i in range(2)]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector([last_t, t]),
        lines=open3d.utility.Vector2iVector([[0, 1]]))
    line_set.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)


if __name__ == '__main__':
    # marker directory
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
    # create ar marker
    dir_name = "armarker"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for i in range(5):
        fileName = "{}/{:02d}.png".format(dir_name, i)
        if not os.path.exists(fileName):
            generator = cv2.aruco.drawMarker(dictionary, id=i, sidePixels=100)
            cv2.imwrite(fileName, generator)

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

    cap = cv2.VideoCapture(1)
    scale = 0.2
    markerLengthMeter = 0.2
    last_t = [0, 0, 0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # show captured frame
        if not show("capture", frame):
            break
        h, w = frame.shape[:2]

        vis.poll_events()
        vis.update_renderer()
        
        # detect markers
        corners, ids, rejectedImgPoints	= cv2.aruco.detectMarkers(frame, dictionary, cameraMatrix=K, distCoeff=distCoeffs)
        if ids is None:
            continue
        if len(ids) <= 0:
            continue
        # draw markers
        drawFrame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # pose estimation
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, markerLengthMeter, K, distCoeffs)
        for i in range(0, len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            drawFrame = cv2.aruco.drawAxis(drawFrame, K, distCoeffs, rvec, tvec, scale)
            R,_ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            t = -R.T @ t
            R = R.T
            # show camera pose on Open3D
            update_camera(K, R, t, scale)
            update_trajectory(last_t, t)
            last_t = t

        # show drawed frame
        if not show("marker", drawFrame):
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

