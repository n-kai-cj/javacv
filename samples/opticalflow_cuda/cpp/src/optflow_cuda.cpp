#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>

int main(int argc, char *argv[])
{
    int width = 640;
    int height = 480;

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.open(0);
    cv::Mat prevgray;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> optflow_cuda = cv::cuda::FarnebackOpticalFlow::create();
    cv::cuda::GpuMat prevgray_cuda;
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

        if (prevgray_cuda.empty())
        {
            prevgray = gray.clone();
            prevgray_cuda.upload(gray);
            continue;
        }

        // calculate dense optical flow
        cv::cuda::GpuMat flow_cuda;
        cv::cuda::GpuMat gray_cuda = cv::cuda::GpuMat(gray);
        optflow_cuda->calc(prevgray_cuda, gray_cuda, flow_cuda);
        cv::Mat flow;
        flow_cuda.download(flow);

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

        prevgray_cuda = gray_cuda;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
