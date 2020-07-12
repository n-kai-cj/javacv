import cv2
import numpy as np

inW = 416
inH = 416
classesFile = "coco.names"
conf = "yolov3.cfg"
model = "yolov3.weights"
confThreshold = 0.5


def postprocess(frame, outs, outLayerType, classes):
    height, width = frame.shape[:2]
    boxes = []
    confidences = []
    classIds = []

    if outLayerType == 'Region':
        for out in outs:
            for i, data in enumerate(out):
                scores = data[5:]
                classId = np.argmax(scores)
                confidence = float(scores[classId])
                if confThreshold < confidence:
                    box = data[0:4] * np.array([width, height, width, height])
                    cx = int(box[0])
                    cy = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    left = int(cx - w / 2)
                    top = int(cy - h / 2)
                    boxes.append([left, top, w, h])
                    confidences.append(confidence)
                    classIds.append(classId)

    # apply non-maximum suppression to suppress weak, overlapping bboxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (left, top, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
            confidence = confidences[i]
            classId = classIds[i]
            # draw bbox rectangle and label on the image
            cv2.rectangle(frame,
                          (left, top),
                          (left+w, top+h),
                          (255,128*float(i),0),2)
            label = "{} {:.2f}".format(classes[classId], confidence)
            labelSize, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.rectangle(frame,
                          (left, top-labelSize[1]),
                          (left+labelSize[0], top+baseline),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label,
                        (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


if __name__ == '__main__':
    print("--- start ---")

    classes = None
    with open(classesFile, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(conf, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    outNames = net.getUnconnectedOutLayersNames()
    outLayers = net.getLayerId(net.getLayerNames()[-1])
    outLayerType = net.getLayer(outLayers).type
    print("read net succeed")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a 4D blob from a frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (inW, inH),
                                     swapRB=True, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(outNames)

        tick_freq = cv2.getTickFrequency()
        t, layersTimings = net.getPerfProfile()
        time_ms = t / tick_freq * 1000.0
        print("inference time: {:.1f}[ms]".format(time_ms))

        # Showing information on the screen
        postprocess(frame, outs, outLayerType, classes)

        cv2.imshow("yolov3", frame)
        key = cv2.waitKey(10)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("--- finish ---")
