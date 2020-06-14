import cv2
import numpy as np

def refineMatches(matches):
    refMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            refMatches.append([m])
    return refMatches

if __name__ == '__main__':
    width = 640
    height = 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # initialize orb and bfmatcher
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    tMat = None
    tKp = None
    tDesc = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect ORB keypoint and compute description
        kp, desc = orb.detectAndCompute(frame, None)
        if tMat is None:
            tMat = frame
            tKp = kp
            tDesc = desc

        # brute force knn match
        matches = bf.knnMatch(desc, tDesc, k=2)

        # apply Lowe's ratio test
        matches = refineMatches(matches)

        # draw matching points
        matchImg = cv2.drawMatchesKnn(frame, kp, tMat, tKp, matches, None)

        cv2.imshow("ORB", matchImg)
        key = cv2.waitKey(10)
        if key == 27: # ESC to exit
            break
        elif key == 32: # Space to update matching
            tMat = None

    cap.release()
    cv2.destroyAllWindows()

