import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.opencv.global.opencv_aruco;
import org.bytedeco.opencv.global.opencv_calib3d;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_aruco.DetectorParameters;
import org.bytedeco.opencv.opencv_aruco.Dictionary;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import java.io.File;

public class ArUco {
    public static void main(String[] args) {
        // marker dictionary
        Dictionary dictionary = opencv_aruco.getPredefinedDictionary(opencv_aruco.DICT_6X6_250);
        File dir_name = new File("armarker");
        if (!dir_name.exists()) {
            dir_name.mkdir();
        }
        for (int i = 0; i < 5; i++) {
            String file_name = String.format("%s/%02d.png", dir_name.getAbsoluteFile().getName(), i);
            if (!new File(file_name).exists()) {
                Mat mat = new Mat();
                opencv_aruco.drawMarker(dictionary, i, 100, mat);
                opencv_imgcodecs.imwrite(file_name, mat);
                releaseMat(mat);
            }
        }

        // Camera parameter
        double fx = 540.627;
        double fy = 550.577;
        double cx = 320.833;
        double cy = 240.796;
        double k1 = 0.1546;
        double k2 = -0.331;
        double p1 = -0.00123;
        double p2 = -0.0001;
    
        Mat K = new Mat(3, 3, opencv_core.CV_64F);
        DoubleIndexer di = K.createIndexer();
        di.put(0, 0, fx);
        di.put(1, 0, 0);
        di.put(2, 0, 0);
        di.put(0, 1, 0);
        di.put(1, 1, fy);
        di.put(2, 1, 0);
        di.put(0, 2, cx);
        di.put(1, 2, cy);
        di.put(2, 2, 1);
        Mat distCoeffs = new Mat(1, 4, opencv_core.CV_64F);
        di = distCoeffs.createIndexer();
        di.put(0, 0, k1);
        di.put(0, 1, k2);
        di.put(0, 2, p1);
        di.put(0, 3, p2);

        VideoCapture cap = new VideoCapture(1);
        Mat _objPoints = new Mat();

        while (true) {
            Mat frame = new Mat();
            if (!cap.read(frame)) {
                break;
            }

            // show captured frame
            OpenCVFX.imshow("capture", frame.clone());
            int key = OpenCVFX.waitKey();
            if (key == 27) {
                break;
            }

            // detect markers
            MatVector markerCorners = new MatVector();
            Mat markerIds = new Mat();
            MatVector rejectedCandidates = new MatVector();
            opencv_aruco.detectMarkers(frame, dictionary, markerCorners, markerIds, DetectorParameters.create(), rejectedCandidates, K, distCoeffs);
            if (markerIds.rows() <= 0) {
                releaseMat(frame, markerIds);
                for (Mat mat : markerCorners.get()) {
                    releaseMat(mat);
                }
                for (Mat mat : rejectedCandidates.get()) {
                    releaseMat(mat);
                }
                continue;
            }

            // draw markers
            Mat drawFrame = frame.clone();
            opencv_aruco.drawDetectedMarkers(drawFrame, markerCorners, markerIds, Scalar.all(255));
            // pose estimation
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            opencv_aruco.estimatePoseSingleMarkers(markerCorners, 0.2f, K, distCoeffs, rvecs, tvecs, _objPoints);
            for (int row = 0; row < rvecs.rows(); row++) {
                Mat rvec = rvecs.row(row);
                Mat tvec = tvecs.row(row);
                opencv_calib3d.drawFrameAxes(drawFrame, K, distCoeffs, rvec, tvec, 0.2f, 3);
                releaseMat(rvec, tvec);
            }
            // show drawed frame
            OpenCVFX.imshow("marker", drawFrame.clone());
            releaseMat(frame, drawFrame, rvecs, tvecs);
        }

        cap.release();
        OpenCVFX.destroyAllWindows();

    }

    private static void releaseMat(Mat... mats) {
        for (Mat mat : mats) {
            if (mat != null) {
                mat.release();
                mat.close();
            }
        }
    }
}