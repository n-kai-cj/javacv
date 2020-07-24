import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_features2d;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_features2d.ORB;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class KMeans {
    public static void main(String[] args) {
        VideoCapture cap = new VideoCapture(0);

        ORB orb = ORB.create(500, 1.2f, 8, 31, 0, 2, ORB.HARRIS_SCORE, 31, 20);
        KeyPointVector keyPointVector = new KeyPointVector();
        Mat grayMat = new Mat();
        Mat bestLabel = new Mat();
        Mat center = new Mat();

        int kmeans_attempts = 10;
        int kmeans_K = 9;

        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            opencv_imgproc.cvtColor(frame, grayMat, opencv_imgproc.COLOR_BGR2GRAY);
            orb.detect(grayMat, keyPointVector);
            if (keyPointVector.size() <= kmeans_K) {
                continue;
            }

            Mat keypointMat = new Mat((int) keyPointVector.size(), 1, opencv_core.CV_32FC2);
            FloatIndexer kpMatIdx = keypointMat.createIndexer();
            for (int i = 0; i < keypointMat.rows(); i++) {
                KeyPoint keyPoint = keyPointVector.get(i);
                kpMatIdx.put(i, keyPoint.pt().x(), keyPoint.pt().y());
            }
            // keypoints clustering with k-means
            opencv_core.kmeans(keypointMat, kmeans_K, bestLabel,
                    new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 10, 1.0),
                    kmeans_attempts, opencv_core.KMEANS_PP_CENTERS, center);

            // draw keypoints
            opencv_features2d.drawKeypoints(frame, keyPointVector, frame);

            // draw kmeans centroid points
            FloatIndexer centerIdx = center.createIndexer();
            Scalar color = new Scalar(255, 0, 255, 0);
            for (int i = 0; i < center.rows(); i++) {
                int cx = (int) centerIdx.get(i, 0);
                int cy = (int) centerIdx.get(i, 1);
                opencv_imgproc.circle(frame, new Point(cx, cy), 6, color, opencv_imgproc.FILLED, opencv_imgproc.CV_AA, 0);
                opencv_imgproc.circle(frame, new Point(cx, cy), 3, Scalar.all(255), opencv_imgproc.FILLED, opencv_imgproc.CV_AA, 0);
                opencv_imgproc.circle(frame, new Point(cx, cy), 1, color, opencv_imgproc.FILLED, opencv_imgproc.CV_AA, 0);
            }

            OpenCVFX.imshow("KMeans", frame);
            if (OpenCVFX.waitKey() == 27) {
                break;
            }

            releaseMat(frame, keypointMat);
        }
        OpenCVFX.destroyAllWindows();
        cap.release();
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
