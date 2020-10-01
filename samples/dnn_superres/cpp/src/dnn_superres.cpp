#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

int main(int argc, char *argv[])
{
    std::string model_name = "FSRCNN_x2.pb";
    std::string algorithm = "fsrcnn";
    int scale = 2;
    bool isGpu = false;

    // dnn superres
    cv::dnn_superres::DnnSuperResImpl sr;
    sr.readModel(model_name);
    sr.setModel(algorithm, scale);
    sr.setPreferableBackend(isGpu ? cv::dnn::DNN_BACKEND_CUDA : cv::dnn::DNN_BACKEND_OPENCV);
    sr.setPreferableTarget(isGpu ? cv::dnn::DNN_TARGET_CUDA : cv::dnn::DNN_TARGET_CPU);

    // video capture
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.open(0);

    while (true)
    {
        cv::Mat frame;
        cv::Mat sup_frame;
        cv::Mat res_frame;

        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        sr.upsample(frame, sup_frame);
        cv::resize(frame, res_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);

        cv::imshow("capture", frame);
        cv::imshow("resize", res_frame);
        cv::imshow("dnn super", sup_frame);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
