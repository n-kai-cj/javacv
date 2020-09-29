import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.opencv.global.opencv_calib3d;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_features2d;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.DMatchVector;
import org.bytedeco.opencv.opencv_core.DMatchVectorVector;
import org.bytedeco.opencv.opencv_core.KeyPointVector;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_features2d.BFMatcher;
import org.bytedeco.opencv.opencv_features2d.ORB;

import java.io.File;

public class MonoVO2 {
    public static void main(String[] args) {
        // initialize orb and bfmatcher
        ORB orb = ORB.create(500, 1.2f, 8, 31, 0, 2, ORB.HARRIS_SCORE, 31, 20);
        BFMatcher bfMatcher = BFMatcher.create(opencv_core.NORM_HAMMING, false);

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

        // OpenGL Viewer
        OpenGlViewer glViewer = new OpenGlViewer(640, 480, "MonoVO2");
        glViewer.setMaxCamNum(-1);
        glViewer.launch();

        File[] imageList = new File("images").listFiles((dir, name) -> name.endsWith("jpg"));
        if (imageList == null) {
            System.err.println("no image files");
            return;
        }

        boolean isFirst = true;
        Mat last_R = new Mat();
        Mat last_t = new Mat();
        Mat mask1 = new Mat();
        for (int i = 0; i < imageList.length - 1; i++) {
            Mat image1 = opencv_imgcodecs.imread(imageList[i].getPath());
            Mat image2 = opencv_imgcodecs.imread(imageList[i + 1].getPath());

            // detect orb keypoint and compute description
            KeyPointVector kp1 = new KeyPointVector();
            KeyPointVector kp2 = new KeyPointVector();
            Mat desc1 = new Mat();
            Mat desc2 = new Mat();
            orb.detectAndCompute(image1, mask1, kp1, desc1);
            orb.detectAndCompute(image2, mask1, kp2, desc2);

            // brute force knn match
            DMatchVectorVector matches = new DMatchVectorVector();
            bfMatcher.knnMatch(desc1, desc2, matches, 2);

            // apply Lowe's test
            matches = refineMatches(matches);

            // draw matching points
            Mat drawImg = new Mat();
            opencv_features2d.drawMatchesKnn(image1, kp1, image2, kp2, matches, drawImg);

            Mat mp1 = new Mat(matches.get().length, 2, opencv_core.CV_64F);
            DoubleIndexer mp1I = mp1.createIndexer();
            Mat mp2 = new Mat(matches.get().length, 2, opencv_core.CV_64F);
            DoubleIndexer mp2I = mp2.createIndexer();
            for (int j = 0; j < matches.get().length; j++) {
                DMatchVector m = matches.get()[j];
                mp1I.put(j, 0, kp1.get()[m.get()[0].queryIdx()].pt().x());
                mp1I.put(j, 1, kp1.get()[m.get()[0].queryIdx()].pt().y());
                mp2I.put(j, 0, kp2.get()[m.get()[0].trainIdx()].pt().x());
                mp2I.put(j, 1, kp2.get()[m.get()[0].trainIdx()].pt().y());
            }

            // calculate essential matrix and decompose to R and t
            Mat R = new Mat();
            Mat t = new Mat();
            Mat eMask = new Mat();
            Mat E = opencv_calib3d.findEssentialMat(mp2, mp1, K, opencv_calib3d.RANSAC, 0.999, 1.0, eMask);
            opencv_calib3d.recoverPose(E, mp2, mp1, K, R, t, eMask);

            if (isFirst) {
                isFirst = false;
                last_R = R.clone();
                last_t = t.clone();
            }

            Mat TwMat = new Mat();
            opencv_core.add(last_t, matMultiply(last_R, t), TwMat);
            Mat RwMat = matMultiply(R, last_R);
            // update estimated current camera pose on OpenGL viewer
            glViewer.addCameraPose(matToFloatArray(RwMat, TwMat));

            // update last trajectories
            last_R = RwMat.clone();
            last_t = TwMat.clone();

            // matching image show
            OpenCVFX.imshow("image", drawImg);
            if (OpenCVFX.waitKey() == 27) {
                break;
            }

            // release mat
            releaseMat(E, R, t, eMask, drawImg, mp1, mp2, TwMat, RwMat, desc1, desc2, image1, image2);
        }

        while (glViewer.isRunning()) {
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

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

    private static DMatchVectorVector refineMatches(DMatchVectorVector matches) {
        DMatchVectorVector refMatches = new DMatchVectorVector();
        for (DMatchVector match : matches.get()) {
            if (match.get()[0].distance() < 0.75 * match.get()[1].distance()) {
                refMatches.push_back(match);
            }
        }
        return refMatches;
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