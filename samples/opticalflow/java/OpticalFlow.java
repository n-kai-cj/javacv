import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_video;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class OpticalFlow {
    public static void main(String[] args) {
        int width = 640;
        int height = 480;
        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, width);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, height);
        cap.open(0);
        Mat prevgray = null;
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            // gray scale
            Mat gray = new Mat();
            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);

            if (prevgray == null || prevgray.empty()) {
                prevgray = gray.clone();
                releaseMat(frame, gray);
                continue;
            }

            // calculate dense optical flow
            Mat flow = new Mat();
            opencv_video.calcOpticalFlowFarneback(prevgray, gray, flow,
                    0.5, 3, 15, 3, 5, 1.1, opencv_video.OPTFLOW_FARNEBACK_GAUSSIAN);

            // draw direction of flow
            FloatIndexer flowatxy = flow.createIndexer();
            for (int j = 0; j < frame.rows(); j += 20) {
                for (int i = 0; i < frame.cols(); i += 20) {
                    float fi = flowatxy.get(j, i, 0);
                    float fj = flowatxy.get(j, i, 1);
                    opencv_imgproc.line(frame, new Point(i, j),
                            new Point((int) (i + fi), (int) (j + fj)),
                            Scalar.all(255), 2, opencv_imgproc.LINE_AA, 0);
                    opencv_imgproc.line(frame, new Point(i, j),
                            new Point((int) (i + fi), (int) (j + fj)),
                            new Scalar(192, 0, 0, 0), 1, opencv_imgproc.LINE_AA, 0);
                    opencv_imgproc.circle(frame, new Point(i, j), 1,
                            Scalar.all(255), 2, opencv_imgproc.LINE_AA, 0);
                    opencv_imgproc.circle(frame, new Point(i, j), 1,
                            new Scalar(0, 133, 255, 0), 1, opencv_imgproc.LINE_AA, 0);
                }
            }

            OpenCVFX.imshow("Dense Optical Flow", frame);
            if (OpenCVFX.waitKey() == 27) {
                break;
            }

            prevgray = gray.clone();

            // release mat
            releaseMat(frame, gray, flow);
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
