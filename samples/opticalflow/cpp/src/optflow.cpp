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
    // some faster than mat image container
    cv::Mat prevgray;
    while (true)
    {
        cv::Mat frame, flow;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // gray scale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (!prevgray.empty())
        {
            // calculate dense optical flow
            cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.4, 1, 12, 2, 8, 1.2, 0);

            for (int y = 0; y < frame.rows; y += 20)
            {
                for (int x = 0; x < frame.cols; x += 20)
                {
                    const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x) * 10;
                    // draw line and flow direction
                    cv::line(frame, cv::Point(x, y), cv::Point(std::round(x + flowatxy.x), std::round(y + flowatxy.y)), cv::Scalar(255, 0, 0));
                    // draw initial point
                    cv::circle(frame, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
                }
            }
        }
        cv::imshow("Dense Optical Flow", frame);
        gray.copyTo(prevgray);

        if (cv::waitKey(10) == 27)
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
