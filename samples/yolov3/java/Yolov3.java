import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.DictValue;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class Yolov3 {
    public static void main(String[] args) {
        int width = 640;
        int height = 480;
        int inpW = 416;
        int inpH = 416;
        double confThreshold = 0.5;
        String cocoNames = "coco.names";
        String configuration = "yolov3.cfg";
        String model = "yolov3.weights";

        Net net = opencv_dnn.readNetFromDarknet(configuration, model);
        ArrayList<String> nameList;
        try {
            nameList = (ArrayList<String>) Files.readAllLines(Paths.get(cocoNames), StandardCharsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }

        IntPointer outLayers = net.getUnconnectedOutLayers();
        String outLayerType = net.getLayer(new DictValue(outLayers.get())).type().getString();
        StringVector outNames = net.getUnconnectedOutLayersNames();

        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, width);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, height);
        cap.open(0);
        while (true) {
            Mat frame = new Mat();
            cap.read(frame);
            if (frame.empty()) {
                break;
            }

            // Create a 4D blob from a frame
            Mat blob = opencv_dnn.blobFromImage(frame, 1 / 255.0,
                    new Size(inpW, inpH), new Scalar(0, 0, 0, 1),
                    true, false, opencv_core.CV_32F);

            // Sets the input to the network
            net.setInput(blob);

            // Runs the forward pass to get outputt of the output layers
            MatVector outs = new MatVector();
            net.forward(outs, outNames);

            DoublePointer layersTimings = new DoublePointer();
            double tick_freq = opencv_core.getTickFrequency();
            double time_ms = net.getPerfProfile(layersTimings) / tick_freq * 1000.0;
            System.out.println(String.format("inference time %.1f[ms]", time_ms));

            // Showing information on the screen
            postprocess(frame, outs, outLayerType, nameList, confThreshold);
            OpenCVFX.imshow("yolov3", frame);

            frame.release();
            frame.close();
        }
        cap.release();
        OpenCVFX.destroyAllWindows();
    }

    private static void postprocess(Mat frame, MatVector outs, String outLayerType, ArrayList<String> nameList, double confThreshold) {
        if (outLayerType.startsWith("Region")) {
            for (int i = 0; i < outs.size(); ++i) {
                Mat o = outs.get(i);
                FloatRawIndexer data = o.createIndexer();
                for (int j = 0; j < o.rows(); ++j) {
                    int cx = (int) (data.get(j, 0) * frame.cols());
                    int cy = (int) (data.get(j, 1) * frame.rows());
                    int w = (int) (data.get(j, 2) * frame.cols());
                    int h = (int) (data.get(j, 3) * frame.rows());
                    Mat scores = o.row(j).colRange(5, o.cols());
                    Point classIdPoint = new Point(1);
                    DoublePointer confidence = new DoublePointer(1);
                    opencv_core.minMaxLoc(scores, null, confidence, null, classIdPoint, null);

                    if (confThreshold < confidence.get()) {
                        int left = cx - w / 2;
                        int top = cy - h / 2;
                        Scalar color = new Scalar(0, 255, (128 * j) % 256, 0);
                        String label = String.format("%s %.2f", nameList.get(classIdPoint.x()), confidence.get());
                        IntPointer baseline = new IntPointer(1);
                        Size labelSize = opencv_imgproc.getTextSize(label, opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseline);
                        opencv_imgproc.rectangle(frame, new Rect(left, top, w, h),
                                color, 2, opencv_imgproc.CV_AA, 0);
                        opencv_imgproc.rectangle(frame, new Point(left, top - labelSize.height()),
                                new Point(left + labelSize.width(), top + baseline.get()),
                                Scalar.all(255), opencv_imgproc.FILLED, opencv_imgproc.CV_AA, 0);
                        opencv_imgproc.putText(frame, label, new Point(left, top), opencv_imgproc.FONT_HERSHEY_SIMPLEX,
                                0.5, new Scalar(0, 0, 0, 0), 1, opencv_imgproc.CV_AA, false);
                    }
                }
            }
        }
    }

}
