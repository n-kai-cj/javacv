import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_features2d;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_cudafeatures2d.DescriptorMatcher;
import org.bytedeco.opencv.opencv_cudafeatures2d.ORB;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class FeatureCuda {
    public static void main(String[] args) {
        int width = 640;
        int height = 480;
        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, width);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, height);
        cap.open(1);

        // initialize orb and bfmatcher
        ORB orb_cuda = ORB.create(500, 1.2f, 8, 31, 0, 2, org.bytedeco.opencv.opencv_features2d.ORB.HARRIS_SCORE, 31, 20, false);
        DescriptorMatcher bfMatcher_cuda = DescriptorMatcher.createBFMatcher(opencv_core.NORM_HAMMING);
        Mat tMat = null;
        KeyPointVector tKp = null;
        GpuMat tDesc_cuda = null;
        GpuMat mask = new GpuMat();
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            opencv_imgproc.cvtColor(frame, frame, opencv_imgproc.COLOR_BGR2GRAY);

            // detect ORB keypoint and compute description
            KeyPointVector kp = new KeyPointVector();
            GpuMat frame_cuda = new GpuMat(frame);
            GpuMat desc_cuda = new GpuMat();
            orb_cuda.detectAndCompute(frame_cuda, mask, kp, desc_cuda);
            if (tMat == null) {
                tMat = frame.clone();
                tKp = kp;
                tDesc_cuda = desc_cuda.clone();
            }

            // brute force knn match
            DMatchVectorVector matches = new DMatchVectorVector();
            bfMatcher_cuda.knnMatch(desc_cuda, tDesc_cuda, matches, 2);

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
                releaseMat(tMat);
                tMat = null;
            }

            releaseMat(frame, matchImg);
            releaseGpuMat(desc_cuda);
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

    private static void releaseGpuMat(GpuMat... mats) {
        for (GpuMat mat : mats) {
            if (mat != null && !mat.isNull()) {
                mat.release();
                mat.close();
            }
        }
    }
}
