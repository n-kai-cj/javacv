#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    cv::Mat img = cv::imread("messi5.jpg");
    cv::Mat template_img = cv::imread("template.jpg");
    int width = template_img.cols;
    int height = template_img.rows;

    cv::Mat res;
    cv::matchTemplate(img, template_img, res, cv::TM_SQDIFF);
    cv::Point top_left;
    cv::minMaxLoc(res, NULL, NULL, &top_left, NULL);
    cv::Point bottom_right(top_left.x + width, top_left.y + height);
    
    cv::rectangle(img, top_left, bottom_right, cv::Scalar(255, 139, 0), 2);
    cv::imshow("TemplateMatching", img);
    while (cv::waitKey(1000) != 27)
    {
        continue;
    }
    return 0;
}
