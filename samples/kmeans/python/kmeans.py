import cv2
import numpy as np


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                         firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    kmeans_attempts = 10
    kmeans_K = 9

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp = orb.detect(gray, None)
        if len(kp) <= kmeans_K:
            continue
        
        data = cv2.KeyPoint_convert(kp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # keypoints clustering with k-means
        ret, label, center = cv2.kmeans(data, kmeans_K, None, criteria, kmeans_attempts, cv2.KMEANS_PP_CENTERS)

        # draw keypoints
        cv2.drawKeypoints(frame, kp, frame)

        # draw kmeans centroid points
        for c in center:
            cv2.circle(frame, (c[0], c[1]), 6, (255,0,255), cv2.FILLED)
            cv2.circle(frame, (c[0], c[1]), 3, (255,255,255), cv2.FILLED)
            cv2.circle(frame, (c[0], c[1]), 1, (255,0,255), cv2.FILLED)

        cv2.imshow("KMeans", frame)
        key = cv2.waitKey(10)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

