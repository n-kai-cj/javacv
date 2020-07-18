#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

int main(int argc, char *argv[])
{
    cv::Mat img = cv::imread("messi5.jpg");
    cv::Mat template_img = cv::imread("template.jpg");
    int width = template_img.cols;
    int height = template_img.rows;

    cv::Ptr<cv::cuda::TemplateMatching> templateMatchingCuda = cv::cuda::createTemplateMatching(template_img.type(), cv::TM_SQDIFF);

    cv::cuda::GpuMat img_cuda(img);
    cv::cuda::GpuMat template_cuda(template_img);
    cv::cuda::GpuMat res_cuda;
    templateMatchingCuda->match(img_cuda, template_cuda, res_cuda);
    cv::Mat res;
    res_cuda.download(res);

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
