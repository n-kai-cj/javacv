import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.GpuMat;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;

public class TemplateMatching {
    public static void main(String[] args) {
        Mat img = opencv_imgcodecs.imread("messi5.jpg");
        Mat template = opencv_imgcodecs.imread("template.jpg");
        int width = template.cols();
        int height = template.rows();
        Mat res = new Mat();
        opencv_imgproc.matchTemplate(img, template, res, opencv_imgproc.TM_SQDIFF);
        Point top_left = new Point();
        opencv_core.minMaxLoc(res, (double[]) null, null, top_left, null, new Mat());
        Point bottom_right = new Point(top_left.x() + width, top_left.y() + height);
        opencv_imgproc.rectangle(img, top_left, bottom_right, new Scalar(255, 139, 0, 0), 2, opencv_imgproc.CV_AA, 0);
        OpenCVFX.imshow("TemplateMatching", img);
        while (OpenCVFX.waitKey() != 27) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        OpenCVFX.destroyAllWindows();
    }

}
