#include <iostream>
#include <opencv2/opencv.hpp>

int width = 640;
int height = 480;

int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.open(0);
    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // gray scale and gaussian blur
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

        // sobel
        cv::Mat sobel, sobelX, sobelY;
        cv::Sobel(gray, sobelX, CV_16S, 1, 0, 3);
        cv::Sobel(gray, sobelY, CV_16S, 0, 1, 3);
        cv::convertScaleAbs(sobelX, sobelX);
        cv::convertScaleAbs(sobelY, sobelY);
        cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);

        // laplacian
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_16S, 3);
        cv::convertScaleAbs(laplacian, laplacian);

        // canny
        cv::Mat canny;
        cv::Canny(gray, canny, 50, 200, 3);

        cv::Mat h1, h2, show;
        cv::hconcat(gray, sobel, h1);
        cv::hconcat(laplacian, canny, h2);
        cv::vconcat(h1, h2, show);

        // show
        cv::imshow("EdgeDetect", show);

        if (cv::waitKey(10) == 27)
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
