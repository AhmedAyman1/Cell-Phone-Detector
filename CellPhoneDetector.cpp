#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;

int main() {
    // Load YOLOv4 model
    String modelConfiguration = "yolov4.cfg";
    String modelWeights = "yolov4.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

    // Load classes names
    std::string classesFile = "coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    std::vector<std::string> classes;
    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }

    // Define the target class name
    std::string targetClassName = "cell phone";

    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Webcam not found or cannot be opened." << std::endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        flip(frame, frame, 0);
        // Perform object detection
        Mat blob = blobFromImage(frame, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        std::vector<Mat> outs;
        std::vector<String> outLayerNames = net.getUnconnectedOutLayersNames();
        net.forward(outs, outLayerNames);

        // Post-process the detection results
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<Rect> boxes;
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int classId = classIdPoint.x;
                if (confidence > 0.5 && classes[classId] == targetClassName) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classId);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        // Apply non-maximum suppression
        std::vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Draw bounding boxes and labels
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            String label = format("%s: %.2f", targetClassName.c_str(), confidences[idx]);
            Rect box = boxes[idx];
            rectangle(frame, box, Scalar(0, 0, 255), 2);
            putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }

        // Show the result
        imshow("Object Detection", frame);

        if (waitKey(1) >= 0) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
