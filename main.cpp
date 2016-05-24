#include <iostream>
#include "Image_segmentation_classifier.h"
#include "Moving_object_tracker.h"

// Higher value = less pixels (faster)
const int RESIZE_FACTOR = 2;
int mode = 1; // car be: TEXTURE = 1 , FEATURE = 2
int nr_of_modes = 2;

// The distance features must move per 1/FRAMERATE second to track
// movement in percentage of the whole frame size
const double MIN_MOVEMENT_THRESHOLD = 1;
double MAX_MAHALANOBIS_DISTANCE = 0.05; //Default value, can be changed manually or by the alorithm

// This factor multiplied with the mean movement vector length gives the euclidian distance threshold
// for features to count as part of the tracked object
// Lower means strickter, a good value is between 0.2 and 0.5
const float MOVEMENT_VECTOR_SIMILARITY_THRESHOLD = 0.3;

int main() {

    // Set up windows:
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

    //Make the image classifier
    Image_segmentation_classifier img_seg_classifier = Image_segmentation_classifier(
            MAX_MAHALANOBIS_DISTANCE);


    bool done;

    while(!done) {

        switch (mode) {
            case 1:

                while (true) {
                    //Get an image from the camera
                    cv::Mat current_image, segmented_image;
                    cap >> current_image;

                    //Classify image
                    img_seg_classifier.segment(current_image, segmented_image);

                    imshow(result_window, segmented_image);

                    int key = cv::waitKey(30);
                    if (key == 'q') {done = true; break;}
                    if (key == 'w') img_seg_classifier.increaseCloseIterations();
                    if (key == 's') img_seg_classifier.decreaseCloseIterations();
                    if (key == 'e') img_seg_classifier.increaseCloseSize();
                    if (key == 'd') img_seg_classifier.decreaseCloseSize();
                    if (key == 'r') img_seg_classifier.increaseMahalanobisDistance();
                    if (key == 'f') img_seg_classifier.decreaseMahalanobisDistance();
                    if (key == 'g') img_seg_classifier.retrain();
                    if (key == 'a') {
                        mode += 1;
                        break;
                    }
                }
                break;

            case 2:

                result_window = "Object tracker";
                cv::namedWindow(result_window);

                cv::Mat raw_image, output_image;
                Moving_object_tracker tracker(400, 10, 1, 0.3, 2);

                // Main loop
                while (true) {
                    // Fetch video stream
                    cap >> raw_image;

                    tracker.track(raw_image, output_image);

                    imshow(result_window, output_image);

                    int key = cv::waitKey(30);
                    if (key == 'q') {done = true; break;}
                    if (key == 'r') tracker.reset();
                    if (key == 'a') {
                        mode = 1;
                        break;
                    }
                }
                break;
        }
    }
}
