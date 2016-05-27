#include <iostream>
#include "Color_model_object_tracker.h"
#include "Moving_object_tracker.h"

// Higher value = less pixels (faster)
const int RESIZE_FACTOR = 2;

//Change this parameter to switch between the different object trackers
int mode = 3; // can be: TEXTURE = 1 , FEATURE = 2, MULTI = 3

// The distance features must move per 1/FRAMERATE second to track movement in percentage of the whole frame size
const double MIN_MOVEMENT_THRESHOLD = 1;

// Default value for the color model
double MAX_MAHALANOBIS_DISTANCE = 0.05;

// This factor multiplied with the mean movement vector length gives the euclidian distance threshold
// for features to count as part of the tracked object
// Lower is stricter, a good value is between 0.2 and 0.5
const float MOVEMENT_VECTOR_SIMILARITY_THRESHOLD = 0.3;

// Returns a point between the two weighted input points
cv::Point2d interpolate_points (
        cv::Point2i pt1, double w1,
        cv::Point2i pt2, double w2)
{
    cv::Point2d result_point;

    result_point.x = (w1*pt1.x + w2*pt2.x) / (w1 + w2);
    result_point.y = (w1*pt1.y + w2*pt2.y) / (w1 + w2);

    return result_point;
}

int main() {

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

    bool done = false;
    while(!done) {

        switch (mode) {

            case 1: {
                // Main window
                std::string result_window;
                cv::namedWindow(result_window);
                cv::moveWindow(result_window, 0, 0);

                //Make the image classifier
                Color_model_object_tracker color_tracker(MAX_MAHALANOBIS_DISTANCE,RESIZE_FACTOR);

                while (true) {
                    //Get an image from the camera
                    cv::Mat current_image, segmented_image;
                    cap >> current_image;

                    //Classify image
                    color_tracker.segment(current_image, segmented_image);
                    cv::resize(segmented_image, segmented_image, cv::Size(640*2, 480*2));
                    imshow(result_window, segmented_image);

                    //Keybindings for å kontrollere modellen
                    int key = cv::waitKey(30);
                    if (key == 'q') { //quit
                        done = true;
                        break;
                    }

                    if (key == 'g') color_tracker.retrain(); // Retrain the model
                    if (key == 'w') color_tracker.increaseRefinementIterations();
                    if (key == 's') color_tracker.decreaseRefinementIterations();
                    if (key == 'e') color_tracker.increaseRefinementKernelSize();
                    if (key == 'd') color_tracker.decreaseRefinementKernelSize();
                    if (key == 'r') color_tracker.increaseMahalanobisDistance();
                    if (key == 'f') color_tracker.decreaseMahalanobisDistance();

                    if (key == 'a') { //Change to the other model
                        mode++;
                        break;
                    }
                }
                break;
            }

            case 2: {

                std::string feature_window = "Detected features";
                cv::namedWindow(feature_window);
                cv::moveWindow(feature_window, 0, 0);

                std::string result_window = "Original features";
                cv::namedWindow(result_window);
                cv::moveWindow(result_window, 0, 0);

                std::string result_window2 = "Additional features";
                cv::namedWindow(result_window2);
                cv::moveWindow(result_window2, 1300, 0);

                cv::Mat raw_image, output_image;

                Moving_object_tracker feature_tracker(400, 10, 10, 0.3, RESIZE_FACTOR);

                // Main loop
                while (true) {

                    // Fetch video stream
                    cap >> raw_image;
                    cv::Mat feature_image, outputImage2;

                    feature_tracker.track(raw_image, feature_image, output_image, outputImage2);

                    cv::resize(feature_image, feature_image, cv::Size(640*2, 480*2));
                    cv::resize(output_image, output_image, cv::Size(640*2, 480*2));
                    cv::resize(outputImage2, outputImage2, cv::Size(640*2, 480*2));

                    imshow(feature_window, feature_image);
                    imshow(result_window, output_image);
                    imshow(result_window2, outputImage2);

                    int key = cv::waitKey(30);
                    if (key == 'q') {
                        done = true;
                        break;
                    }

                    if (key == 'g') feature_tracker.reset();
                    if (key == 'x') feature_tracker.wipe_rectangle_model();

                    if (key == 'a') {
                        mode++;
                        break;
                    }
                }
                break;
            }

            case 3: {

                std::string feature_window = "Feature tracker";
                std::string color_model_window = "Color tracker";
                std::string final_window = "Final tracker";

                cv::namedWindow(feature_window);
                cv::namedWindow(color_model_window);
                cv::namedWindow(final_window);

                cv::moveWindow(feature_window, 0, 0);
                cv::moveWindow(color_model_window, 1900, 0);
                cv::moveWindow(final_window, 1000, 0);

                // Make the trackers
                Color_model_object_tracker color_tracker(MAX_MAHALANOBIS_DISTANCE,RESIZE_FACTOR);
                Moving_object_tracker feature_tracker(400, 10, 10, 0.3, RESIZE_FACTOR);

                bool trained_color_models = false;

                // Main loop
                while (true) {

                    // Fetch video stream
                    cv::Mat raw_image, segmented_image, feature_image, final_image, trash_image;
                    cap >> raw_image;

                    // FEATURES
                    feature_tracker.track(raw_image, trash_image, feature_image, trash_image);
                    cv::resize(feature_image, feature_image, cv::Size(640*2, 480*2));
                    imshow(feature_window, feature_image);

                    // COLOR
                    color_tracker.segment(raw_image, segmented_image);
                    cv::resize(segmented_image, segmented_image, cv::Size(640*2, 480*2));
                    imshow(color_model_window, segmented_image);

                    // If the feature detector has detected the object, train the color detector
                    if (!trained_color_models && feature_tracker.found_object()) {
                        color_tracker.train(feature_tracker.get_object_image_lab());
                        trained_color_models = true;
                    }

                    // Find weighted crosshair from the two trackers
                    cv::Point2i color_model_crosshair_pos = color_tracker.get_object_position();
                    cv::Point2i feature_crosshair_pos = feature_tracker.get_object_position();
                    double color_model_confidence = color_tracker.get_confidence_value();
                    double feature_confidence = feature_tracker.get_confidence_value();

                    cv::Point2d crosshair_position = interpolate_points(color_model_crosshair_pos, color_model_confidence,
                                                                        feature_crosshair_pos, feature_confidence);

                    //Show the final image with weighted crosshair
                    raw_image.copyTo(final_image);
                    cv::drawMarker(final_image, crosshair_position, cv::Scalar::all(255));
                    cv::resize(final_image, final_image, cv::Size(640*2, 480*2));
                    imshow(final_window, final_image);

                    printf("Feature confidence: %.2f, color confidence: %.2f\n",
                           feature_tracker.get_confidence_value(), color_tracker.get_confidence_value());

                    int key = cv::waitKey(30);
                    if (key == 'q') {
                        done = true;
                        break;
                    }

                    if (key == 'w') color_tracker.increaseRefinementIterations();
                    if (key == 's') color_tracker.decreaseRefinementIterations();
                    if (key == 'e') color_tracker.increaseRefinementKernelSize();
                    if (key == 'd') color_tracker.decreaseRefinementKernelSize();
                    if (key == 'r') color_tracker.increaseMahalanobisDistance();
                    if (key == 'f') color_tracker.decreaseMahalanobisDistance();

                    if (key == 'x') feature_tracker.wipe_rectangle_model();
                    if (key == 'g') {
                        trained_color_models = false;
                        feature_tracker.reset();
                    }

                    if (key == 'a') {
                        mode = 1;
                        break;
                    }
                }
                break;
            }
        }
    }
}
