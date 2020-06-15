#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    int width = 640;
    int height = 480;

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.open(0);
    cv::Mat prevgray;
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // gray scale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (prevgray.empty())
        {
            prevgray = gray.clone();
            continue;
        }

        // calculate dense optical flow
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prevgray, gray, flow,
                                     0.5, 3, 15, 3, 5, 1.1, cv::OPTFLOW_FARNEBACK_GAUSSIAN);

        // draw direction of flow
        for (int y = 0; y < frame.rows; y += 20)
        {
            for (int x = 0; x < frame.cols; x += 20)
            {
                const cv::Point2f flowatxy = flow.at<cv::Point2f>(y, x);
                cv::line(frame, cv::Point(x, y),
                         cv::Point((int)std::round(x + flowatxy.x), (int)std::round(y + flowatxy.y)),
                         cv::Scalar::all(255), 2, cv::LINE_AA);
                cv::line(frame, cv::Point(x, y),
                         cv::Point((int)std::round(x + flowatxy.x), (int)std::round(y + flowatxy.y)),
                         cv::Scalar(192, 0, 0), 1, cv::LINE_AA);
                cv::circle(frame, cv::Point(x, y), 1,
                           cv::Scalar::all(255), 2, cv::LINE_AA);
                cv::circle(frame, cv::Point(x, y), 1,
                           cv::Scalar(0, 133, 255), 1, cv::LINE_AA);
            }
        }

        cv::imshow("Dense Optical Flow", frame);

        if (cv::waitKey(10) == 27)
        {
            break;
        }

        prevgray = gray.clone();
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
