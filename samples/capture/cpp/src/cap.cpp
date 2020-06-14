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
        cv::imshow("capture", frame);
        if (cv::waitKey(10) == 27)
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
