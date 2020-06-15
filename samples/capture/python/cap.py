import cv2

width = 640
height = 480
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.open(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
