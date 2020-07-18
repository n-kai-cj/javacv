import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_cudaimgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.GpuMat;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_cudaimgproc.TemplateMatching;

public class TemplateMatchingCuda {
    public static void main(String[] args) {
        Loader.load(opencv_java.class);
        Mat img = opencv_imgcodecs.imread("messi5.jpg");
        Mat template = opencv_imgcodecs.imread("template.jpg");
        int width = template.cols();
        int height = template.rows();

        TemplateMatching templateMatchingCuda = opencv_cudaimgproc.createTemplateMatching(template.type(), opencv_imgproc.TM_SQDIFF);
        GpuMat img_cuda = new GpuMat(img);
        GpuMat template_cuda = new GpuMat(template);
        GpuMat res_cuda = new GpuMat();
        templateMatchingCuda.match(img_cuda, template_cuda, res_cuda);
        Mat res = new Mat();
        res_cuda.download(res);

        Point top_left = new Point();
        opencv_core.minMaxLoc(res, (double[]) null, null, top_left, null, new Mat());
        Point bottom_right = new Point(top_left.x() + width, top_left.y() + height);
        opencv_imgproc.rectangle(img, top_left, bottom_right, new Scalar(255, 139, 0, 0), 2, opencv_imgproc.CV_AA, 0);
        OpenCVFX.imshow("TemplateMatchingCuda", img);
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
