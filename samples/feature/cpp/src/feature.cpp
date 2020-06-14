#include <iostream>
#include <opencv2/opencv.hpp>

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
    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(500, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    cv::Mat tMat;
    std::vector<cv::KeyPoint> tKp;
    cv::Mat tDesc;
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        // detect ORB keypoint and compute description
        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        orb->detectAndCompute(frame, cv::Mat(), kp, desc);
        if (tMat.empty())
        {
            tMat = frame.clone();
            tKp = kp;
            tDesc = desc.clone();
        }

        // brute force knn match
        std::vector<std::vector<cv::DMatch>> matches;
        bfMatcher->knnMatch(desc, tDesc, matches, 2);

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
