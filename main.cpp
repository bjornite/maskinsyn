#include <cmath>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "Moving_object_tracker.h"

using namespace std;

// Higher value = less pixels (faster)
const int RESIZE_FACTOR = 2;

// The distance features must move per 1/FRAMERATE second to track
// movement in percentage of the whole frame size
const double MIN_MOVEMENT_THRESHOLD = 1;

// This factor multiplied with the mean movement vector length gives the euclidian distance threshold
// for features to count as part of the tracked object
// Lower means strickter, a good value is between 0.2 and 0.5
const float MOVEMENT_VECTOR_SIMILARITY_THRESHOLD = 0.3;

int main() {

    // Set up windows:
    std::string result_window = "Object tracker";
    cv::namedWindow(result_window);

    // Get video from webcam or internal camera
    cv::VideoCapture cap;
    cap.open(1);
    if (!cap.isOpened())
    {
        cap.open(0);

        if (!cap.isOpened())
        {
            printf("Could not detect any camera, exiting...");
            return -1;
        }
    }

    // Setting frame rate
    cap.set(CV_CAP_PROP_FPS, 5);

    cv::Mat raw_image, output_image;
    Moving_object_tracker tracker(400, 10, 1, 0.3, 2);

    // Main loop
    while (true) {
        // Fetch video stream
        cap >> raw_image;

        tracker.track(raw_image, output_image);

        imshow(result_window, output_image);

        int key = cv::waitKey(30);
        if (key == 'q') break;
        if (key == 'r') tracker.reset();
    }
}