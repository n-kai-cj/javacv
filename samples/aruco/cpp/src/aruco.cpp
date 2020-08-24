#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>

bool checkFileExistence(const std::string &str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
}

int main(int argc, char *argv[])
{
    // marker dictionary
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    std::string dir_name = "armarker";

    // make markers
    if (!checkFileExistence(dir_name))
    {
        std::filesystem::create_directory(dir_name);
    }
    for (int i = 0; i < 5; i++)
    {
        char buf[100];
        snprintf(buf, sizeof(buf), "%s/%02d.png", dir_name, i);
        std::string fileName = buf;
        if (!checkFileExistence(fileName))
        {
            cv::Mat markerImage;
            cv::aruco::drawMarker(dictionary, i, 100, markerImage);
            cv::imwrite(fileName, markerImage);
        }
    }

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

    cv::VideoCapture cap = cv::VideoCapture(0);

    while (true)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            break;
        }
        
        // show captured frame
        cv::imshow("capture", frame);
        if (cv::waitKey(1) == 27)
        {
            break;
        }

        // detect markers
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        if (markerIds.size() <= 0)
        {
            continue;
        }

        // draw markers
        cv::Mat drawFrame = frame.clone();
        cv::aruco::drawDetectedMarkers(drawFrame, markerCorners, markerIds);
        // pose estimation
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.2, K, distCoeffs, rvecs, tvecs);
        for (int i = 0; i < rvecs.size(); i++)
        {
            cv::Vec3d rvec = rvecs[i];
            cv::Vec3d tvec = tvecs[i];
            cv::drawFrameAxes(drawFrame, K, distCoeffs, rvec, tvec, 0.2, 3);
        }

        // show drawed frame
        cv::imshow("marker", drawFrame);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
