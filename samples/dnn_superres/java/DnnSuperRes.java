import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn_superres.DnnSuperResImpl;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class DnnSuperRes {
    public static void main(String[] args) {
        String model_name = "FSRCNN_x2.pb";
        String algorithm = "fsrcnn";
        int scale = 2;
        boolean isGpu = false;

        // dnn superres
        DnnSuperResImpl sr = new DnnSuperResImpl();
        sr.readModel(model_name);
        sr.setModel(algorithm, scale);
        sr.setPreferableBackend(isGpu ? opencv_dnn.DNN_BACKEND_CUDA : opencv_dnn.DNN_BACKEND_OPENCV);
        sr.setPreferableTarget(isGpu ? opencv_dnn.DNN_TARGET_CUDA : opencv_dnn.DNN_TARGET_CPU);

        // video capture
        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, 640);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, 480);
        cap.open(0);
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            Mat sup_frame = new Mat();
            sr.upsample(frame, sup_frame);
            Mat res_frame = new Mat();
            int w = frame.cols();
            int h = frame.rows();
            opencv_imgproc.resize(frame, res_frame, new Size(w*scale, h*scale), scale, scale, opencv_imgproc.INTER_LINEAR);

            OpenCVFX.imshow("capture", frame);
            OpenCVFX.imshow("resize", res_frame);
            OpenCVFX.imshow("dnn super", sup_frame);
            if (OpenCVFX.waitKey() == 27) {
                break;
            }
        }
        cap.release();
        OpenCVFX.destroyAllWindows();
    }

}
