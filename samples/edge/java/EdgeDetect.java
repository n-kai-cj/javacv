import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_videoio;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

public class EdgeDetect {
    public static void main(String[] args) {
        int width = 640;
        int height = 480;
        VideoCapture cap = new VideoCapture();
        cap.set(opencv_videoio.CAP_PROP_FRAME_WIDTH, width);
        cap.set(opencv_videoio.CAP_PROP_FRAME_HEIGHT, height);
        cap.open(0);
        while (true) {
            Mat frame = new Mat();
            boolean ret = cap.read(frame);
            if (!ret) break;

            // gray scale and gaussian blur
            Mat gray = new Mat();
            opencv_imgproc.cvtColor(frame, gray, opencv_imgproc.COLOR_BGR2GRAY);
            opencv_imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0, 0, opencv_core.BORDER_DEFAULT);

            // sobel
            Mat sobel = new Mat();
            Mat sobelX = new Mat();
            Mat sobelY = new Mat();
            opencv_imgproc.Sobel(gray, sobelX, opencv_core.CV_16S, 1, 0, 3, 1, 0, opencv_core.BORDER_DEFAULT);
            opencv_imgproc.Sobel(gray, sobelY, opencv_core.CV_16S, 0, 1, 3, 1, 0, opencv_core.BORDER_DEFAULT);
            opencv_core.convertScaleAbs(sobelX, sobelX);
            opencv_core.convertScaleAbs(sobelY, sobelY);
            opencv_core.addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);

            // laplacian
            Mat laplacian = new Mat();
            Mat laplacianGrad = new Mat();
            opencv_imgproc.Laplacian(gray, laplacianGrad, opencv_core.CV_16S, 3, 1, 0, opencv_core.BORDER_DEFAULT);
            opencv_core.convertScaleAbs(laplacianGrad, laplacian);

            // canny
            Mat canny = new Mat();
            opencv_imgproc.Canny(gray, canny, 50, 200, 3, false);

            Mat h1 = new Mat();
            Mat h2 = new Mat();
            Mat show = new Mat();
            opencv_core.hconcat(new MatVector(gray, sobel), h1);
            opencv_core.hconcat(new MatVector(laplacian, canny), h2);
            opencv_core.vconcat(new MatVector(h1, h2), show);

            // show
            OpenCVFX.imshow("EdgeDetect", show);
            // "Esc" to exit
            if (OpenCVFX.waitKey() == 27) {
                break;
            }

            // release mat
            releaseMat(
                    show, h1, h2,
                    canny,
                    laplacianGrad, laplacian,
                    sobelX, sobelY, sobel,
                    gray, frame);
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
