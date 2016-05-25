#include <iostream>
#include "Image_segmentation_classifier.h"
#include "Moving_object_tracker.h"

// Higher value = less pixels (faster)
const int RESIZE_FACTOR = 2;
const int mode = 2; // car be: TEXTURE = 1 , FEATURE = 2

// The distance features must move per 1/FRAMERATE second to track
// movement in percentage of the whole frame size
const double MIN_MOVEMENT_THRESHOLD = 1;

// This factor multiplied with the mean movement vector length gives the euclidian distance threshold
// for features to count as part of the tracked object
// Lower means strickter, a good value is between 0.2 and 0.5
const float MOVEMENT_VECTOR_SIMILARITY_THRESHOLD = 0.3;

int main() {

    // Main window
    std::string result_window;

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

    switch (mode) {
        case 1:

            result_window = "Color segmented image";
            cv::namedWindow(result_window);

            while (true) {
                //Get an image from the camera
                cv::Mat current_image, segmented_image;
                cap >> current_image;

                //Make the image classifier
                Image_segmentation_classifier img_seg_classifier = Image_segmentation_classifier();

                //Classify image
                img_seg_classifier.segment(current_image, segmented_image);

                imshow(result_window, segmented_image);

                int key = cv::waitKey(30);
                if (key == 'q') break;
            }
            break;

        case 2:

            std::string feature_window = "Detected features";
            cv::namedWindow(feature_window);
            cv::moveWindow(feature_window, 0, 0);

            result_window = "Original features";
            cv::namedWindow(result_window);
            cv::moveWindow(result_window, 0, 0);

            std::string result_window2 = "Additional features";
            cv::namedWindow(result_window2);
            cv::moveWindow(result_window2, 1300, 0);
        

            cv::Mat raw_image, output_image;
            Moving_object_tracker tracker(400, 10, 10, 0.3, 2);
  

            // Main loop
            while (true) {
                // Fetch video stream
                cap >> raw_image;

                cv::Mat feature_image, outputImage2;

                tracker.track(raw_image, feature_image, output_image, outputImage2);
                
                imshow(feature_window, feature_image);
                imshow(result_window, output_image);
                imshow(result_window2, outputImage2);
                
                
                int key = cv::waitKey(30);
                if (key == 'q') break;
                if (key == 'r') tracker.reset();
                if (key == 'w') tracker.wipe_rectangle_model();
            }
            break;
    }
}
