#include <iostream>
#include <string>
#include <fstream>
#include <thread>

#include <cassert>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>

inline void draw_line(const float x1, const float y1, const float z1,
                      const float x2, const float y2, const float z2)
{
    glBegin(GL_LINES);
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
    glEnd();
}

inline void draw_rectangle(const float x, const float y, const float z)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBegin(GL_POLYGON);
    glVertex3f(x, y, z);
    glVertex3f(-x, y, z);
    glVertex3f(-x, -y, z);
    glVertex3f(x, -y, z);
    glEnd();
}

inline void draw_frustum(const float w)
{
    const float h = w * 0.75f;
    const float z = w * 1.25f;

    glColor3f(0.9f, 0.7f, 0.7f);
    draw_rectangle(w, h, z);

    glColor3f(0.8f, 0.2f, 0.2f);
    glLineWidth(1.0);
    draw_line(0.0f, 0.0f, 0.0f, w, h, z);
    draw_line(0.0f, 0.0f, 0.0f, w, -h, z);
    draw_line(0.0f, 0.0f, 0.0f, -w, -h, z);
    draw_line(0.0f, 0.0f, 0.0f, -w, h, z);
    draw_line(w, h, z, w, -h, z);
    draw_line(-w, h, z, -w, -h, z);
    draw_line(-w, h, z, w, h, z);
    draw_line(-w, -h, z, w, -h, z);
}

void draw_camera(const pangolin::OpenGlMatrix &gl_cam_pose, const float width)
{
    glPushMatrix();
    glMultMatrixd(gl_cam_pose.m);
    draw_frustum(width);
    glPopMatrix();
}

void draw_trajectory(pangolin::OpenGlMatrix last_t, pangolin::OpenGlMatrix t)
{
    glPushMatrix();
    glColor3f(0.8f, 0.2f, 0.2f);
    glLineWidth(1.0);
    draw_line((float)last_t.m[12], (float)last_t.m[13], (float)last_t.m[14], (float)t.m[12], (float)t.m[13], (float)t.m[14]);
    glPopMatrix();
}

void update_camera(std::vector<pangolin::OpenGlMatrix> cams, const float width)
{
    if (cams.size() <= 0)
        return;

    // draw each cameras and trajectory lines
    draw_camera(cams[0], width);
    for (int i = 0; i < cams.size() - 1; ++i)
    {
        auto c1 = cams[i];
        auto c2 = cams[i + 1];
        draw_camera(c2, width);
        draw_trajectory(c1, c2);
    }
}

void draw_axis()
{
    // draw axis
    glPushMatrix();
    glLineWidth(5.0);
    float size = 1;
    glColor3f(1, 0, 0);
    draw_line(0, 0, 0, size, 0, 0);
    glColor3f(0, 1, 0);
    draw_line(0, 0, 0, 0, size, 0);
    glColor3f(0, 0, 1);
    draw_line(0, 0, 0, 0, 0, size);
    glPopMatrix();
}

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

    // create opengl window
    pangolin::CreateWindowAndBind("MonoVO2", 640, 480);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // depth testing to be enabled for 3D mouse handler
    glEnable(GL_DEPTH_TEST);
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));
    // Create Interactive View in window
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / (float)480.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));
    pangolin::OpenGlMatrix cam_pose;
    cam_pose.SetIdentity();
    std::vector<pangolin::OpenGlMatrix> cams;

    std::string image_dir = "images/";
    std::vector<std::string> fsList = getFileList(image_dir);

    cv::Mat last_t, last_R;
    bool first = true;

    for (int i = 0; i < fsList.size() - 1 && !pangolin::ShouldQuit(); i++)
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Activate efficiently by object
        d_cam.Activate(s_cam);

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
        // set camera pose
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                cam_pose.m[row + col * 4] = Rw.at<double>(row, col);
            }
            cam_pose.m[(row + 1) * 4 - 1] = 0.0;
        }
        for (int row = 0; row < 3; ++row)
        {
            cam_pose.m[12 + row] = Tw.at<double>(row, 0);
        }
        cam_pose.m[15] = 1.0;
        cams.push_back(cam_pose);

        // update last trajectories
        last_t = Tw.clone();
        last_R = Rw.clone();

        cv::imshow("image", drawImg);
        if (cv::waitKey(1) == 27)
        {
            break;
        }

        // draw camera frustum of pyramid
        update_camera(cams, (const float)0.3);
        draw_axis();
        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    while (!pangolin::ShouldQuit())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Activate efficiently by object
        d_cam.Activate(s_cam);
        update_camera(cams, 0.3);
        pangolin::FinishFrame();
    }

    cv::destroyAllWindows();
    return 0;
}
