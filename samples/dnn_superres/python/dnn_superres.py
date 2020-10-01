import cv2

model_name = "FSRCNN_x2.pb"
algorithm = "fsrcnn"
scale = 2
isGpu = False

# dnn superres
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_name)
sr.setModel(algorithm, scale)
sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA if isGpu else cv2.dnn.DNN_BACKEND_OPENCV)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA if isGpu else cv2.dnn.DNN_TARGET_OPENCV)

# video capture
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.open(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]
    sup_frame = sr.upsample(frame)
    res_frame = cv2.resize(frame, (w*scale, h*scale), cv2.INTER_LINEAR)
    cv2.imshow("capture", frame)
    cv2.imshow("resize", res_frame)
    cv2.imshow("dnn super", sup_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
