#include <iostream>
#include <string>
#include <fstream>
#include <thread>

#include <cassert>
#include <filesystem>

#include <opencv2/opencv.hpp>

bool checkFileExistence(const std::string &str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
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

std::vector<std::string> getFileList(std::string path)
{
    std::vector<std::string> list;
    for (auto &entry : std::filesystem::directory_iterator(path))
    {
        list.push_back(entry.path().filename().string());
    }
    return list;
}

int main(int argc, char *argv[])
{
    // initailize orb and bfmatcher
    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);

    // Camera parameter
    double fx = 540.627;
    double fy = 550.577;
    double cx = 320.833;
    double cy = 240.796;
    double k1 = 0.1546;
    double k2 = -0.331;
    double p1 = -0.00123;
    double p2 = -0.0001;
    cv::Mat K = (cv::Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat1d(1, 4) << k1, k2, p1, p2);

    std::string image_dir = "images/";
    std::vector<std::string> fsList = getFileList(image_dir);

    cv::Mat last_t, last_R;
    bool first = true;

    for (int i = 0; i < fsList.size() - 1; i++)
    {
        cv::Mat image1 = cv::imread(image_dir + fsList[i]);
        cv::Mat image2 = cv::imread(image_dir + fsList[i + 1]);

        // detect ORB keypoint and compute description
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        std::vector<std::vector<cv::DMatch>> matches;
        orb->detectAndCompute(image1, cv::Mat(), kp1, desc1);
        orb->detectAndCompute(image2, cv::Mat(), kp2, desc2);

        // brute force knn match
        bfMatcher->knnMatch(desc1, desc2, matches, 2);

        // apply Lowe's test
        matches = refineMatches(matches);

        // draw match
        cv::Mat drawImg;
        cv::drawMatches(image1, kp1, image2, kp2, matches, drawImg);

        std::vector<cv::Point2f> mp1, mp2;
        for (auto m : matches)
        {
            mp1.push_back(kp1[m[0].queryIdx].pt);
            mp2.push_back(kp2[m[0].trainIdx].pt);
        }

        // calculate essential matrix and decompose to R and t
        cv::Mat E, R, t, mask;
        E = cv::findEssentialMat(mp2, mp1, K, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, mp2, mp1, K, R, t, mask);
        if (first)
        {
            first = false;
            last_t = t.clone();
            last_R = R.clone();
        }
        cv::Mat Tw = last_t + last_R * t;
        cv::Mat Rw = R * last_R;
        std::cout << Rw << std::endl << Tw << std::endl;

        // update last trajectories
        last_t = Tw.clone();
        last_R = Rw.clone();

        cv::imshow("image", drawImg);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
