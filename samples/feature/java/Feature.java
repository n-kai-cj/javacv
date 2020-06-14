import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_features2d;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.BFMatcher;
import org.bytedeco.opencv.opencv_features2d.ORB;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class Feature {
    public static void main(String[] args) {
        int width = 640;
        int height = 480;
        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, width);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, height);
        cap.open(0);

        // initialize orb and bfmatcher
        ORB orb = ORB.create(500, 1.2f, 8, 31, 0, 2, ORB.HARRIS_SCORE, 31, 20);
        BFMatcher bfMatcher = BFMatcher.create(opencv_core.NORM_HAMMING, false);
        Mat tMat = null;
        KeyPointVector tKp = null;
        Mat tDesc = null;
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            // detect ORB keypoint and compute description
            KeyPointVector kp = new KeyPointVector();
            Mat desc = new Mat();
            Mat mask = new Mat();
            orb.detectAndCompute(frame, mask, kp, desc);
            if (tMat == null) {
                tMat = frame.clone();
                tKp = kp;
                tDesc = desc.clone();
            }

            // brute force knn match
            DMatchVectorVector matches = new DMatchVectorVector();
            bfMatcher.knnMatch(desc, tDesc, matches, 2);

            // apply Lowe's ratio test
            matches = refineMatches(matches);

            // draw matching points
            Mat matchImg = new Mat();
            opencv_features2d.drawMatchesKnn(frame, kp, tMat, tKp, matches, matchImg);

            OpenCVFX.imshow("ORB", matchImg);
            int key = OpenCVFX.waitKey();
            if (key == 27) { // ESC to exit
                break;
            } else if (key == 32) { // Space to update matching
                releaseMat(tMat, tDesc);
                tMat = null;
            }

            releaseMat(frame, desc, mask, matchImg);
        }
        cap.release();
        OpenCVFX.destroyAllWindows();
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
