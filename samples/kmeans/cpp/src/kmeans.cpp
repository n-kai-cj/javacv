#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    cv::VideoCapture cap;
    cap.open(0);

    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(500, 1.2, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    int kmeans_attempts = 10;
    int kmeans_K = 9;

    while (true)
    {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypointVector;
        orb->detect(gray, keypointVector);
        if (keypointVector.size() <= kmeans_attempts)
        {
            continue;
        }

        cv::Mat kpMat(keypointVector.size(), 1, CV_32FC2);
        
        for (int i = 0; i < keypointVector.size(); i++)
        {
            cv::KeyPoint kp = keypointVector[i];
            kpMat.at<cv::Point2f>(i) = cv::Point2f(kp.pt.x, kp.pt.y);
        }

        // keypoints clustering with k-means
		cv::Mat bestLabel, center;
        cv::kmeans(kpMat, kmeans_K, bestLabel,
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
            kmeans_attempts, cv::KMEANS_PP_CENTERS, center);

        // draw keypoints
        cv::drawKeypoints(frame, keypointVector, frame);

        // draw kmeans centroid points
        cv::Scalar color(255, 0, 255, 0);
        for (int i = 0; i < center.rows; i++)
        {
            cv::Point2f cp = center.at<cv::Point2f>(i);
            cv::circle(frame, cp, 6, color, cv::FILLED);
            cv::circle(frame, cp, 3, cv::Scalar::all(255), cv::FILLED);
            cv::circle(frame, cp, 1, color, cv::FILLED);
        }

        cv::imshow("KMeans", frame);
        int key = cv::waitKey(10);
        if (key == 27)
        { // ESC to exit
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

