import cv2
import numpy as np

width = 640
height = 480
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.open(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # gray scale and gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # sobel
    sobelX = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobelY = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sobelX = cv2.convertScaleAbs(sobelX)
    sobelY = cv2.convertScaleAbs(sobelY)
    sobel = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

    # laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)

    # canny
    canny = cv2.Canny(gray, 50, 200, 3)

    # show
    h1 = cv2.hconcat((gray, sobel))
    h2 = cv2.hconcat((laplacian, canny))
    show = cv2.vconcat((h1, h2))

    cv2.imshow("EdgeDetect", show)

    key = cv2.waitKey(10)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
