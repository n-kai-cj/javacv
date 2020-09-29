import os
import time
import glob
import collections

import cv2
import numpy as np

import open3d

def update_camera(K, w, h, R, t, vis, scale=0.3):
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


def update_trajectory(last_t, t, vis, color=[0.8, 0.2, 0.2]):
    colors = [color for i in range(2)]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector([last_t, t]),
        lines=open3d.utility.Vector2iVector([[0, 1]]))
    line_set.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)


def refineMatches(matches):
    refMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            refMatches.append([m])
    return refMatches


if __name__ == '__main__':
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

    # initialize Open3D
    vis = open3d.visualization.Visualizer()
    vis.create_window(
        window_name='MonoVO2',
        width=640,
        height=480,
        left=100,
        top=100
    )
    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    vis.add_geometry(axis)

    # initialize orb and bfmatcher
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                         firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    isFirst = True
    last_t = [0, 0, 0]
    last_R = np.identity(3)
    last_R[0,0] = 1
    last_R[1,1] = 1
    last_R[2,2] = 1
    # input file list
    image_list = 'images/*.jpg'
    flist = [name for name in sorted(glob.glob(image_list))]
    
    for i in range(0, len(flist)-1):
        vis.poll_events()
        vis.update_renderer()

        image1 = cv2.imread(flist[i])
        image2 = cv2.imread(flist[i+1])
        h, w = image1.shape[:2]
        # detect ORB keypoint and compute description
        kp1, desc1 = orb.detectAndCompute(image1, None)
        kp2, desc2 = orb.detectAndCompute(image2, None)
        # brute force knn match
        matches = bf.knnMatch(desc1, desc2, k=2)
        matches = refineMatches(matches)
        # draw match
        drawImg = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None)
        mp1 = []
        mp2 = []
        for m in matches:
            mp1.append(kp1[m[0].queryIdx].pt)
            mp2.append(kp2[m[0].trainIdx].pt)
        mp1 = np.array([mp1]).reshape(len(mp1), 2)
        mp2 = np.array([mp2]).reshape(len(mp2), 2)
        # Calculate Essential Matrix and Decompose to R,t
        E, e_inliers = cv2.findEssentialMat(points1=mp2, points2=mp1, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inliercnt, R, t, inliers = cv2.recoverPose(E, mp2, mp1, K)
        t = t.flatten()
        if isFirst:
            isFirst = False
            last_R = R
            last_t = t

        mulMat = last_R @ t
        Tw  = last_t + (last_R @ t)
        Rw = R @ last_R

        # show camera pose on Open3D
        update_camera(K, w, h, Rw, Tw, vis)
        update_trajectory(last_t, Tw, vis)

        # update last trajectories
        last_t = Tw
        last_R = Rw

        cv2.imshow("image", drawImg)
        if cv2.waitKey(1) == 27:
            break

    try:
        while True:
            time.sleep(0.01)
            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    vis.destroy_window()
