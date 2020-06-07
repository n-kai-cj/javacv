import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class VCapture {
    public static void main(String[] args) {
        VideoCapture cap = new VideoCapture(0);
        int count = 0;
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            OpenCVFX.imshow("JavaFX Capture", frame);

            if (count++ > 100) {
                break;
            }

            frame.release();
            frame.close();
        }
        OpenCVFX.destroyAllWindows();
        cap.release();
    }
}
