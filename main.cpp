
#include <iostream>
#include "Image_segmentation_classifier.h"

const int mode = 1; // car be: TEXTURE = 1 , FEATURE = 2

int main() {

    // Set up windows:
    std::string matches_win = "Matching features";
    cv::namedWindow(matches_win);

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

            while(true) {
                //Get an image from the camera
                cv::Mat current_image, segmented_image;
                cap >> current_image;

                //Make the image classifier
                Image_segmentation_classifier img_seg_classifier = Image_segmentation_classifier();

                //Classify image
                img_seg_classifier.segment(current_image, segmented_image);

                imshow(matches_win, segmented_image);

                int key = cv::waitKey(30);
                if (key == 'q') break;
            }
            break;

        case 2:
            break;
    }
}