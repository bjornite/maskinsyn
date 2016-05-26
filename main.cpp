#include <iostream>
#include "Color_model_object_tracker.h"
#include "Moving_object_tracker.h"

// Higher value = less pixels (faster)
const int RESIZE_FACTOR = 1;

//Change this parameter to switch between the different object trackers
int mode = 1; // car be: TEXTURE = 1 , FEATURE = 2


int nr_of_modes = 2;

// The distance features must move per 1/FRAMERATE second to track
// movement in percentage of the whole frame size
const double MIN_MOVEMENT_THRESHOLD = 1;
double MAX_MAHALANOBIS_DISTANCE = 0.05; //Default value for the color model

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


    //Make the image classifier
    Color_model_object_tracker color_model_object_tracker = Color_model_object_tracker(
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
                    color_model_object_tracker.segment(current_image, segmented_image);

                    imshow(result_window, segmented_image);

                    //Keybindings for å kontrollere modellen
                    int key = cv::waitKey(30);
                    if (key == 'q') { //quit
                        done = true;
                        break;
                    }
                    if (key == 'g') color_model_object_tracker.retrain(); //Retrain the model
                    if (key == 'w') color_model_object_tracker.increaseCloseIterations();
                    if (key == 's') color_model_object_tracker.decreaseCloseIterations();
                    if (key == 'e') color_model_object_tracker.increaseCloseSize();
                    if (key == 'd') color_model_object_tracker.decreaseCloseSize();
                    if (key == 'r') color_model_object_tracker.increaseMahalanobisDistance();
                    if (key == 'f') color_model_object_tracker.decreaseMahalanobisDistance();
                    
                    if (key == 'a') { //Change to the other model
                        mode += 1;
                        break;
                    }
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
                    if (key == 'q') {done = true; break;}
                    if (key == 'r') tracker.reset();
                    if (key == 'w') tracker.wipe_rectangle_model();
                    if (key == 'a') {mode = 1; break;}
                }
                break;
        }
    }
}
