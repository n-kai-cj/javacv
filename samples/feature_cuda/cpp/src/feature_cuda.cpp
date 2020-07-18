#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>

std::vector<std::vector<cv::DMatch>> refineMatches(std::vector<std::vector<cv::DMatch>> matches);

int main(int argc, char *argv[])
{
    int width = 640;
    int height = 480;
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.open(0);

    // initailize orb and bfmatcher
    cv::Ptr<cv::FeatureDetector> orb_cuda = cv::cuda::ORB::create(500, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20, false);
    cv::Ptr<cv::cuda::DescriptorMatcher> bfMatcher_cuda = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    cv::Mat tMat;
    std::vector<cv::KeyPoint> tKp;
    cv::cuda::GpuMat tDesc_cuda;
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // detect ORB keypoint and compute description
        std::vector<cv::KeyPoint> kp;
        std::vector<std::vector<cv::DMatch>> matches;
        cv::cuda::GpuMat desc;
        cv::cuda::GpuMat frame_cuda(frame);
        orb_cuda->detectAndCompute(frame_cuda, cv::cuda::GpuMat(), kp, desc);
        if (tMat.empty())
        {
            tMat = frame.clone();
            tKp = kp;
            tDesc_cuda = desc.clone();
        }

        // brute force knn match
        bfMatcher_cuda->knnMatch(desc, tDesc_cuda, matches, 2);

        // apply Lowe's ratio test
        matches = refineMatches(matches);

        // draw matching points
        cv::Mat matchImg;
        cv::drawMatches(frame, kp, tMat, tKp, matches, matchImg);

        cv::imshow("ORB", matchImg);
        int key = cv::waitKey(10);
        if (key == 27)
        { // ESC to exit
            break;
        }
        else if (key == 32)
        { // Space to update matching
            tMat.release();
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

std::vector<std::vector<cv::DMatch>> refineMatches(std::vector<std::vector<cv::DMatch>> matches)
{
    std::vector<std::vector<cv::DMatch>> refMatches;
    for (auto match : matches)
    {
        if (match[0].distance < 0.75 * match[1].distance)
        {
            refMatches.push_back(match);
        }
    }
    return refMatches;
}
