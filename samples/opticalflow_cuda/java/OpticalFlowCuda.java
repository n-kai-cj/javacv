import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.GpuMat;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_cudaoptflow.FarnebackOpticalFlow;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class OpticalFlowCuda {
    public static void main(String[] args) {
        int width = 640;
        int height = 480;
        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, width);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, height);
        cap.open(0);
        GpuMat prevgray_cuda = null;
        FarnebackOpticalFlow optflow_cuda = FarnebackOpticalFlow.create();
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            // gray scale
            Mat gray = new Mat();
            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);

            if (prevgray_cuda == null || prevgray_cuda.empty()) {
                prevgray_cuda = new GpuMat(gray);
                releaseMat(frame, gray);
                continue;
            }

            // calculate dense optical flow
            GpuMat gray_cuda = new GpuMat(gray);
            GpuMat flow_cuda = new GpuMat();
            Mat flow = new Mat();
            optflow_cuda.calc(prevgray_cuda, gray_cuda, flow_cuda);
            flow_cuda.download(flow);

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

            prevgray_cuda = gray_cuda;

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

    private static void releaseGpuMat(GpuMat... mats) {
        for (GpuMat mat : mats) {
            if (mat != null) {
                mat.release();
                mat.close();
            }
        }
    }
}
