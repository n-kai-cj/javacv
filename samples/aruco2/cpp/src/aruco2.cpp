#include <iostream>
#include <string>
#include <fstream>

#include <cassert>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

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

int main(int argc, char *argv[])
{
    // marker dictionary
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h10);

    // create ar marker
    std::string dir_name = "armarker";
    if (!checkFileExistence(dir_name))
    {
        std::filesystem::create_directory(dir_name);
    }
    for (int i = 0; i < 5; i++)
    {
        char buf[100];
        snprintf(buf, sizeof(buf), "%s/%02d.png", dir_name.c_str(), i);
        std::string fileName = buf;
        if (!checkFileExistence(fileName))
        {
            cv::Mat markerImage;
            cv::aruco::drawMarker(dictionary, i, 100, markerImage, 1);
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

    // create opengl window
    pangolin::CreateWindowAndBind("ArUco2", 640, 480);
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

    cv::VideoCapture cap = cv::VideoCapture(1);
    float scale = 0.2f;
    float markerLengthMeter = 0.2f;
    int camSize = 10;

    while (!pangolin::ShouldQuit())
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Activate efficiently by object
        d_cam.Activate(s_cam);

        cv::Mat frame;
        cap.read(frame);
        int w = frame.cols;
        int h = frame.rows;

        cv::imshow("capture", frame);
        int key = cv::waitKey(1);
        if (key == 27)
        {
            break;
        }

        // detect markers
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        if (markerIds.size() > 0)
        {
            // draw markers
            cv::Mat drawFrame = frame.clone();
            cv::aruco::drawDetectedMarkers(drawFrame, markerCorners, markerIds);
            // pose estimation
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, markerLengthMeter, K, distCoeffs, rvecs, tvecs);
            for (int i = 0; i < rvecs.size(); i++)
            {
                cv::Vec3d rvec = rvecs[i];
                cv::Vec3d tvec = tvecs[i];
                cv::aruco::drawAxis(drawFrame, K, distCoeffs, rvec, tvec, scale);

                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Mat t(3, 1, CV_64F);
                for (int row = 0; row < t.rows; ++row)
                {
                    t.at<double>(row, 0) = tvec[row];
                }
                cv::transpose(R, R);
                t = -R * t;
                // set camera pose
                for (int row = 0; row < 3; ++row)
                {
                    for (int col = 0; col < 3; ++col)
                    {
                        cam_pose.m[row + col * 4] = R.at<double>(row, col);
                    }
                    cam_pose.m[(row + 1) * 4 - 1] = 0.0;
                }
                for (int row = 0; row < 3; ++row)
                {
                    cam_pose.m[12 + row] = t.at<double>(row, 0);
                }
                cam_pose.m[15] = 1.0;
                cams.push_back(cam_pose);
                if (cams.size() > camSize)
                {
                    cams.erase(cams.begin());
                }

                // show drawed frame
                cv::imshow("marker", drawFrame);
                key = cv::waitKey(1);
            }
        }
        // draw camera frustum of pyramid
        update_camera(cams, (const float)0.3);
        draw_axis();
        // Swap frames and Process Events
        pangolin::FinishFrame();

        if (key == 27)
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
