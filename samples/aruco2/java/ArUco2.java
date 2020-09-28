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

public class ArUco2 {
    public static void main(String[] args) {
        // marker dictionary
        Dictionary dictionary = opencv_aruco.getPredefinedDictionary(opencv_aruco.DICT_APRILTAG_36h10);
        File dir = new File("armarker");
        if (!dir.exists()) {
            dir.mkdir();
        }
        for (int i = 0; i < 5; i++) {
            String file_name = String.format("%s/%02d.png", dir.getAbsoluteFile().getName(), i);
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

        VideoCapture cap = new VideoCapture(0);
        float scale = 0.2f;
        float markerLengthMeter = 0.2f;
        Mat _objPoints = new Mat();

        // OpenGL Viewer
        OpenGlViewer glViewer = new OpenGlViewer(640, 480, "ArUco2");
        glViewer.setMaxCamNum(10);
        glViewer.launch();

        while (glViewer.isLaunch()) {
            Mat frame = new Mat();
            if (!cap.read(frame)) {
                break;
            }

            // show capture frame
            OpenCVFX.imshow("capture", frame.clone());
            int key = OpenCVFX.waitKey();
            if (key == 27) {
                break;
            }

            MatVector markerCorners = new MatVector();
            Mat markerIds = new Mat();
            MatVector rejectedCandidates = new MatVector();
            // detect markers
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
            opencv_aruco.drawDetectedMarkers(drawFrame, markerCorners, markerIds, new Scalar(138, 255, 0, 0));
            // pose estimation
            Mat rvecs = new Mat();
            Mat tvecs = new Mat();
            opencv_aruco.estimatePoseSingleMarkers(markerCorners, markerLengthMeter, K, distCoeffs, rvecs, tvecs, _objPoints);
            for (int row = 0; row < rvecs.rows(); row++) {
                Mat rvec = rvecs.row(row);
                Mat tvec = tvecs.row(row);
                Mat rmat = new Mat();
                // rvec[3,1](rx, ry, rz) to rmat[3,3]
                opencv_calib3d.Rodrigues(rvec, rmat);

                Mat t = new Mat(3, 1, opencv_core.CV_64F);
                DoubleIndexer tveci = tvec.createIndexer();
                DoubleIndexer ti = t.createIndexer();
                for (int i = 0; i < t.rows(); i++) {
                    // multiply negative value to matrix conversion
                    ti.put(i, 0, -1 * tveci.get(0, 0, i));
                }
                Mat Rw = new Mat();
                opencv_core.transpose(rmat, Rw);
                Mat Tw = matMultiply(Rw, t);
                // update estimated current camera pose on OpenGL viewer
                glViewer.addCameraPose(matToFloatArray(Rw, Tw));
                // draw axis on image
                opencv_calib3d.drawFrameAxes(drawFrame, K, distCoeffs, rvec, tvec, scale, 3);
                // show draw frame
                OpenCVFX.imshow("marker", drawFrame.clone());
                key = OpenCVFX.waitKey();
                releaseMat(rvec, rmat, tvec, t, Rw, Tw);
            }
            if (key == 27) {
                break;
            }
            releaseMat(frame, drawFrame, rvecs, tvecs);
        }
        cap.release();
        OpenCVFX.destroyAllWindows();
        glViewer.destroy();
    }

    private static float[] matToFloatArray(Mat R, Mat t) {
        float[] ret = new float[16];
        DoubleIndexer ri = R.createIndexer();
        DoubleIndexer ti = t.createIndexer();
        for (int row = 0; row < R.rows(); ++row) {
            for (int col = 0; col < R.cols(); ++col) {
                ret[row + col * 4] = (float) ri.get(row, col);
            }
            ret[(row + 1) * 4 - 1] = 0.0f;
        }
        for (int row = 0; row < t.rows(); ++row) {
            ret[12 + row] = (float) ti.get(row, 0);
        }
        ret[15] = 1.0f;
        return ret;
    }

    private static Mat matMultiply(Mat left, Mat right) {
        Mat ret = new Mat(left.rows(), right.cols(), opencv_core.CV_64F);
        DoubleIndexer reti = ret.createIndexer();
        DoubleIndexer li = left.createIndexer();
        DoubleIndexer ri = right.createIndexer();
        for (int row = 0; row < left.rows(); ++row) {
            for (int col = 0; col < right.cols(); ++col) {
                double s = 0;
                for (int i = 0; i < left.cols(); i++) {
                    s += li.get(row, i) * ri.get(i, col);
                }
                reti.put(row, col, s);
            }
        }
        return ret;
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