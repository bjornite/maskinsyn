#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Homography_estimation.h"
#include "Image_segmentation_classifier.h"

using namespace std;

const int mode = 1; // car be: TEXTURE = 1 , FEATURE = 2

// Higher value = less pixels (faster)
const int resize_factor = 1;

// The distance features must move per 1/FRAMERATE second to track
// movement in percentage of the whole frame size
const double min_movement_percentage = 0.5;

const int image_width = 640;
const int image_height = 480;


const cv::Size resized_size(image_width / resize_factor, image_height / resize_factor);
const float min_pixel_movement = ((image_width / resize_factor) / 100 ) * min_movement_percentage;

const cv::Matx33d Camera_Matrix{632.29863082751251, 0, 319.5, 0, 632.29863082751251, 239.5, 0, 0, 1};

const cv::Mat Distortion_Coefficients =
        (cv::Mat_<double>(5,1) << 0.070528331223347215, 0.26247385180956367, 0, 0, -1.0640942232949715);


std::vector<cv::DMatch> extract_good_ratio_matches(
        const std::vector<std::vector<cv::DMatch>>& matches, double max_ratio)
{
    std::vector<cv::DMatch> good_ratio_matches;

    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < matches[i][1].distance * max_ratio)
            good_ratio_matches.push_back(matches[i][0]);
    }

    return good_ratio_matches;
}

// Extracts matched points from matches into keypts1 and 2
void extract_matching_points(
        const std::vector<cv::KeyPoint>& keypts1, const std::vector<cv::KeyPoint>& keypts2,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Point2f>& matched_pts1, std::vector<cv::Point2f>& matched_pts2)
{
    matched_pts1.clear();
    matched_pts2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        matched_pts1.push_back(keypts1[matches[i].queryIdx].pt);
        matched_pts2.push_back(keypts2[matches[i].trainIdx].pt);
    }
}

// Returns the euclidian distance between two points in 2D-space
float euclidian_distance(
        cv::Point2f pt1,
        cv::Point2f pt2)
{
   return sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
}

// Fills a binary mask given the distance between matching points
void mask_stationary_features(
        const std::vector<cv::Point2f>& matched_pts1,
        const std::vector<cv::Point2f>& matched_pts2,
        std::vector<char>& mask)
{
    mask.clear();

    for (int i = 0; i < matched_pts1.size(); i++) {
        mask.push_back(euclidian_distance(matched_pts1.at(i), matched_pts2.at(i)) > min_pixel_movement);
    }
}


// Masks matches and returns a vector with the matches corresponding to true-entries in the mask
void get_unmasked_keypoints (
        const std::vector<cv::DMatch>& matches,
        const std::vector<char>& mask,
        const std::vector<cv::KeyPoint>& keypoints,
        std::vector<cv::KeyPoint>& unmasked_keypoints)
{
    if (matches.size() != mask.size()){
        CV_Error(cv::Error::StsBadSize,"matches and mask must be the same size");
    }
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            unmasked_keypoints.push_back(keypoints.at(matches.at(i).queryIdx));
        }
    }
}

// Updates crosshair position to be the mean of the given keypoints
cv::Point2d calculate_crosshair_position (
        const std::vector<cv::KeyPoint> &keypoints)
{
    double mean_x = 0;
    double mean_y = 0;

    for (int i = 0; i < keypoints.size(); i++)
    {
        mean_x += keypoints.at(i).pt.x;
        mean_y += keypoints.at(i).pt.y;
    }

    mean_x /= keypoints.size();
    mean_y /= keypoints.size();

    return cv::Point2d(mean_x, mean_y);
}



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

            cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

            // Current and previous image pointers
            cv::Mat previous_image, current_image, object_vis;

            std::vector<cv::KeyPoint> last_keypoints;
            cv::Mat last_descriptors;

            cap >> previous_image;
            cap >> object_vis;
            detector->detect( previous_image, last_keypoints);
            detector->compute(previous_image, last_keypoints, last_descriptors);

            // Setting up crosshair image
            cv::Mat crosshair_image;
            cv::Point2d crosshair_position(0, 0);

            bool saved_object = false;
            cv::Mat saved_object_descriptors;
            std::vector<cv::KeyPoint> saved_object_features;
            cv::Mat object_reference_image;

            // Main loop
            while(true) {
                cv::Mat raw_image;

                cap >> raw_image;

                // Undistorting raw_image into current image
                //cv::undistort(raw_image, current_image, Camera_Matrix, Distortion_Coefficients);
                //current_image = raw_image;

                // Make it grayscale
                cv::Mat grayscale_image;
                cv::cvtColor(raw_image, grayscale_image, cv::COLOR_BGR2GRAY);

                // Make it smaller to save computation power
                resize(grayscale_image, current_image, resized_size, 0, 0, cv::INTER_LINEAR);

                // Copy image to crosshair image
                current_image.copyTo(crosshair_image);

                std::vector<cv::KeyPoint> current_keypoints;
                detector->detect(current_image, current_keypoints);

                cv::Mat current_descriptors;

                cv::BFMatcher matcher{detector->defaultNorm()};

                //-- Draw keypoints
                cv::Mat feature_vis;


                //cv::drawKeypoints(current_image, current_keypoints, feature_vis, cv::Scalar{0,255,0});
                std::vector<char> mask;
                std::vector<cv::KeyPoint> moving_features;


                //Only look for matches if we have some features to compare
                if (!last_descriptors.empty()) {

                    std::vector<std::vector<cv::DMatch>> matches;
                    std::vector<std::vector<cv::DMatch>> object_matches;

                    detector->compute(current_image, current_keypoints, current_descriptors);
                    matcher.knnMatch(current_descriptors, last_descriptors, matches, 2);

                    std::vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.5);

                    // Only update crosshair position if there is a decent number of matching features
                    if (good_matches.size() >= 10) {

                        std::vector<cv::Point2f> matching_pts1;
                        std::vector<cv::Point2f> matching_pts2;

                        // Find matching features
                        extract_matching_points(current_keypoints, last_keypoints,
                                                good_matches, matching_pts1, matching_pts2);

                        // Mask features that are not moving
                        mask_stationary_features(matching_pts1, matching_pts2, mask);

                        // Get the moving features
                        get_unmasked_keypoints(good_matches, mask, current_keypoints, moving_features);

                        if (moving_features.size() > 10) {
                            // Updating crosshair position to be mean of moving features
                            crosshair_position = calculate_crosshair_position(moving_features);
                            //printf("%f\n%f\n\n", crosshair_position.x, crosshair_position.y);

                            //Compute descriptors for the moving features. This can be optimized by looking them up. Currently computes these twice.
                            if (!saved_object) {
                                detector->compute(current_image, moving_features, saved_object_descriptors);
                                saved_object_features = moving_features;
                                current_image.copyTo(object_reference_image);
                                saved_object = true;
                                printf("Object saved!\n");
                                printf("Saved %d features\n", (int) moving_features.size());
                            }
                        }
                    }

                    if (saved_object) {
                        //look for saved features in the image
                        matcher.knnMatch(current_descriptors, saved_object_descriptors, object_matches, 2);

                        std::vector<cv::DMatch> good_object_matches = extract_good_ratio_matches(object_matches, 0.5);

                        if (good_object_matches.size() > 5) {

                            cv::drawMatches(current_image, current_keypoints, object_reference_image,
                                            saved_object_features,
                                            good_object_matches, object_vis);

                            printf("found object!\n");
                        }
                        //Draw them
                        //Update the crosshair position
                    }

                    // Draw moving features
                    cv::drawKeypoints(crosshair_image, moving_features, crosshair_image);

                    //cv::drawMatches(current_image, current_keypoints, previous_image, last_keypoints, good_matches, feature_vis,-1,-1,mask);
                    //cv::drawMatches(current_image, current_keypoints, previous_image, last_keypoints, moving_features, feature_vis);
                }


                // Draw the Crosshair
                cv::drawMarker(crosshair_image, crosshair_position, cv::Scalar::all(255), cv::MARKER_CROSS, 100, 2, 8);

                //-- Show detected (drawn) matches
                cv::Mat final_image;
                resize(crosshair_image, final_image, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
                //imshow(matches_win, final_image);
                imshow(matches_win, object_vis);

                previous_image = current_image;
                last_keypoints = current_keypoints;
                last_descriptors = current_descriptors;


                int key = cv::waitKey(30);
                if (key == 'q') break;
                if (key == 'r') saved_object = false;
            }
    }
}