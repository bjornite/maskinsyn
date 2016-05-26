#include <iostream>
#include "Image_segmentation_classifier.h"
#include "Moving_object_tracker.h"

// Higher value = less pixels (faster)
const int RESIZE_FACTOR = 2;
int mode = 1; // car be: TEXTURE = 1 , FEATURE = 2, MULTI = 3
int nr_of_modes = 3;

// The distance features must move per 1/FRAMERATE second to track
// movement in percentage of the whole frame size
const double MIN_MOVEMENT_THRESHOLD = 1;
double MAX_MAHALANOBIS_DISTANCE = 0.05; //Default value, can be changed manually or by the alorithm

// This factor multiplied with the mean movement vector length gives the euclidian distance threshold
// for features to count as part of the tracked object
// Lower means strickter, a good value is between 0.2 and 0.5
const float MOVEMENT_VECTOR_SIMILARITY_THRESHOLD = 0.3;

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
    Image_segmentation_classifier img_seg_classifier = Image_segmentation_classifier(
            MAX_MAHALANOBIS_DISTANCE);

    bool done;

    while(!done) {

        switch (mode) {
            case 1: {

                while (true) {
                    //Get an image from the camera
                    cv::Mat current_image, segmented_image;
                    cap >> current_image;

                    //Classify image
                    img_seg_classifier.segment(current_image, segmented_image);

                    imshow(result_window, segmented_image);

                    int key = cv::waitKey(30);
                    if (key == 'q') {
                        done = true;
                        break;
                    }
                    if (key == 'w') img_seg_classifier.increaseCloseIterations();
                    if (key == 's') img_seg_classifier.decreaseCloseIterations();
                    if (key == 'e') img_seg_classifier.increaseCloseSize();
                    if (key == 'd') img_seg_classifier.decreaseCloseSize();
                    if (key == 'r') img_seg_classifier.increaseMahalanobisDistance();
                    if (key == 'f') img_seg_classifier.decreaseMahalanobisDistance();
                    if (key == 'g') img_seg_classifier.retrain();
                    if (key == 'a') {
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
                    if (key == 'r') tracker.reset();
                    if (key == 'w') tracker.wipe_rectangle_model();
                    if (key == 'a') {
                        mode++;
                        break;
                    }
                }
                break;
            }

            case 3: {

                std::string feature_window = "Feature tracker";
                std::string texture_window = "Texture tracker";
                std::string final_window = "Final tracker";

                cv::namedWindow(feature_window);
                cv::namedWindow(texture_window);
                cv::namedWindow(final_window);

                cv::moveWindow(feature_window, 0, 0);
                cv::moveWindow(texture_window, 1300, 0);
                cv::moveWindow(final_window, 650, 0);

                Moving_object_tracker tracker(400, 10, 10, 0.3, 2);

                bool trained_textures = false;

                // Main loop
                while (true) {

                    // Fetch video stream
                    cv::Mat raw_image, segmented_image, feature_image, final_image, trash_image;
                    cap >> raw_image;

                    // FEATURES
                    tracker.track(raw_image, trash_image, feature_image, trash_image);
                    cv::resize(feature_image, feature_image, cv::Size(640*2, 480*2));
                    imshow(feature_window, feature_image);

                    // TEXTURE
                    //Classify image
                    img_seg_classifier.segment(raw_image, segmented_image);
                    imshow(texture_window, segmented_image);

                    // If the feature detector has detected the object, train the texture detector
                    if (!trained_textures && tracker.found_object()) {
                        img_seg_classifier.train(tracker.get_object_image_lab());
                    }

                    // Find weighted crosshair from the two trackers
                    cv::Point2i texture_crosshair_pos = img_seg_classifier.get_object_position();
                    cv::Point2i feature_crosshair_pos = tracker.get_object_position();
                    double texture_confidence = img_seg_classifier.get_confidence_value();
                    double feature_confidence = tracker.get_confidence_value();

                    cv::Point2d crosshair_position = interpolate_points(texture_crosshair_pos, texture_confidence,
                                                                        feature_crosshair_pos, feature_confidence);

                    raw_image.copyTo(final_image);
                    cv::drawMarker(final_image, crosshair_position, cv::Scalar::all(255));
                    cv::resize(final_image, final_image, cv::Size(640*2, 480*2));
                    imshow(final_window, final_image);

                    printf("Feature confidence: %.2f, Texture confidence: %.2f\n",
                           tracker.get_confidence_value(), img_seg_classifier.get_confidence_value());

                    int key = cv::waitKey(30);
                    if (key == 'q') {
                        done = true;
                        break;
                    }

                    if (key == 'w') img_seg_classifier.increaseCloseIterations();
                    if (key == 's') img_seg_classifier.decreaseCloseIterations();
                    if (key == 'e') img_seg_classifier.increaseCloseSize();
                    if (key == 'd') img_seg_classifier.decreaseCloseSize();
                    if (key == 'r') img_seg_classifier.increaseMahalanobisDistance();
                    if (key == 'f') img_seg_classifier.decreaseMahalanobisDistance();
                    if (key == 'r') {
                        trained_textures = false;
                        tracker.reset();
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
