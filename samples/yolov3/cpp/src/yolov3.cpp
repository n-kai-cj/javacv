#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Prompt Declare
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net net, std::vector<std::string> classes);

int width = 640;
int height = 480;
int inW = 416;
int inH = 416;
std::string classesFile = "coco.names";
std::string conf = "yolov3.cfg";
std::string model = "yolov3.weights";
float confThreshold = 0.5;

int main(int argc, char* argv[]) {
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.open(0);

    std::ifstream ifs(classesFile.c_str());
    std::vector<std::string> classes;
    std::string line;
    while (std::getline(ifs, line)) classes.push_back(line);

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(conf, model);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Create a 4D blob from a frame
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inW, inH), cv::Scalar(0,0,0), true, false);

        // Sets the input to the netwrok
        net.setInput(blob);

        //  Runs the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;
        net.forward(outs, outNames);

        std::vector<double> layersTimings;
        double tick_freq = cv::getTickFrequency();
        double time_ms = net.getPerfProfile(layersTimings) / tick_freq * 1000.0;
        //fprintf(stdout, "inference time %.1f[ms]\n", time_ms);

        // Showing information on the screen
        postprocess(frame, outs, net, classes);

        cv::imshow("yolov3", frame);
        if (cv::waitKey(30) == 27) {
            break;
        }
        
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net net, std::vector<std::string> classes) {
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    if (outLayerType == "Region") {
        for (cv::Mat out : outs) {
            float* data = (float*)out.data;
            // retrieve each detected objects
            for (int i = 0; i < out.rows; ++i, data+=out.cols) {
                int cx = (int)(data[0] * frame.cols);
                int cy = (int)(data[1] * frame.rows);
                int w = (int)(data[2] * frame.cols);
                int h = (int)(data[3] * frame.rows);
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point classIdPoint;
                double confidence;

                cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                if (confThreshold < confidence) {
                    int left = cx - w / 2;
                    int top = cy - h / 2;
                    cv::rectangle(frame, cv::Rect(left, top, w, h), cv::Scalar(255,(128*i)%256,0), 2);
                    std::string label = cv::format("%s %.2f", classes[classIdPoint.x].c_str(), confidence);
                    int baseline;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

                    top = std::max(top, labelSize.height);
                    cv::rectangle(frame, cv::Point(left, top-labelSize.height),
                        cv::Point(left + labelSize.width, top + baseline), cv::Scalar::all(255), cv::FILLED);
                    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
                }
            }
        }
    }
}
