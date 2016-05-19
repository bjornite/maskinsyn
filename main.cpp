#include <cmath>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Homography_estimation.h"

using namespace cv;
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

const int image_width = 640;
const int image_height = 480;

const Size resized_size(image_width / RESIZE_FACTOR, image_height / RESIZE_FACTOR);
const float min_pixel_movement = ((image_width / RESIZE_FACTOR) / 100 ) * MIN_MOVEMENT_THRESHOLD;

// x1, x2, y1, y2
int object_boundary[4];

cv::Point2d drawed_mean_vector;

std::vector<cv::DMatch> extract_good_ratio_matches(
        const std::vector<std::vector<cv::DMatch>>& matches,
        double max_ratio)
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
        const std::vector<cv::KeyPoint>& keypts1,
        const std::vector<cv::KeyPoint>& keypts2,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Point2f>& matched_pts1,
        std::vector<cv::Point2f>& matched_pts2)
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

// Updates a binary mask by removing hipsters (points moving differently from the mean)
void mask_false_moving_features (
        const std::vector<cv::Point2f>& matched_pts1,
        const std::vector<cv::Point2f>& matched_pts2,
        std::vector<char>& mask)
{
    // Find mean vector
    float direction[2] = {0, 0};
    int unmasked_cnt = 0;

    // Only use non-masked keypoints
    for (int i = 0; i < matched_pts1.size(); i++)
    {
        if (mask.at(i) == true) {
            direction[0] += matched_pts2.at(i).x - matched_pts1.at(i).x;
            direction[1] += matched_pts2.at(i).y - matched_pts1.at(i).y;
            unmasked_cnt++;
        }

    }

    direction[0] /= unmasked_cnt;
    direction[1] /= unmasked_cnt;

    cv::Point2d mean_vector(direction[0], direction[1]);
    drawed_mean_vector = mean_vector;
    float mean_direction = atan2f(direction[1], direction[0]) * (180 / M_PI);
    //printf("Mean angle: %d\n", (int)mean_direction);

    // Find minimum alloved euclidian distance from the mean vector
    float mean_vector_length = sqrt(pow(direction[0], 2) + pow(direction[1], 2));
    float minimum_distance = mean_vector_length * MOVEMENT_VECTOR_SIMILARITY_THRESHOLD;

    // Mask keypoints with euclidian distance from mean greater than MOVEMENT_VECTOR_SIMILARITY_THRESHOLD
    for (int i = 0; i < matched_pts1.size(); i++)
    {
        // Only look at non-masked keypoints
        if (mask.at(i) == true) {
            // Create vector from point1 to point2
            int x = matched_pts2.at(i).x - matched_pts1.at(i).x;
            int y = matched_pts2.at(i).y - matched_pts1.at(i).y;
            cv::Point2d p(x, y);
            float a = euclidian_distance(p, mean_vector);
            float b = minimum_distance;

            // Mask points with greater euclidian distance to mean vector than threshold
            if (euclidian_distance(p, mean_vector) > minimum_distance) {
                mask.at(i) = false;
            }
        }
    }
}

// Masks matches and returns a vector with the matches corresponding to true-entries in the mask
void get_unmasked_keypoints (
        const std::vector<cv::DMatch>& matches,
        const std::vector<char>& mask,
        const std::vector<KeyPoint>& keypoints,
        std::vector<cv::KeyPoint>& unmasked_keypoints)
{
    if (matches.size() != mask.size()){
        CV_Error(Error::StsBadSize,"matches and mask must be the same size");
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

// Updates the object boundary (rectangle)
void update_object_boundary (
     int object_boundary[],
     vector<KeyPoint>& keyPoints)
{
    int minX = keyPoints.at(0).pt.x;
    int maxX = keyPoints.at(0).pt.x;
    int minY = keyPoints.at(0).pt.y;
    int maxY = keyPoints.at(0).pt.y;

    for (int i = 0; i < keyPoints.size(); i++)
    {
        if (keyPoints.at(i).pt.x < minX) minX = keyPoints.at(i).pt.x;
        if (keyPoints.at(i).pt.x > maxX) maxX = keyPoints.at(i).pt.x;
        if (keyPoints.at(i).pt.y < minY) minY = keyPoints.at(i).pt.y;
        if (keyPoints.at(i).pt.y > maxY) maxY = keyPoints.at(i).pt.y;
    }

    object_boundary[0] = minX;
    object_boundary[1] = maxX;
    object_boundary[2] = minY;
    object_boundary[3] = maxY;
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

    // Current and previous image pointers
    cv::Mat previous_image, current_image, object_vis;

    // Initialize feature detector
    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);
    std::vector<KeyPoint> previous_keypoints;
    cv::Mat previous_descriptors;

    // Fetch video
    cap >> previous_image;
    cap >> object_vis;

    // Find keypoints and their descriptors for the first image
    detector->detect( previous_image, previous_keypoints);
    detector->compute(previous_image, previous_keypoints, previous_descriptors);

    // Debug
    cv::Mat row = previous_descriptors.row(0);

    // Set up crosshair image
    cv::Mat crosshair_image;
    cv::Point2d crosshair_position(0, 0);

    // Set up rectangle
    cv::Point2d rectangle_pt1;
    cv::Point2d rectangle_pt2;

    // The model of the object
    bool saved_object = false;
    cv::Mat saved_object_descriptors;
    std::vector<KeyPoint> saved_object_features;
    cv::Mat object_reference_image;

    // Main loop
    while(true) {

        // Fetch video stream
        cv::Mat raw_image;
        cap >> raw_image;

        // Make image smaller to save computation power
        resize(raw_image, current_image, resized_size, 0, 0, INTER_LINEAR);

        // Copy image to crosshair image
        current_image.copyTo(crosshair_image);

        std::vector<KeyPoint> current_keypoints;
        detector->detect( current_image, current_keypoints);

        cv::Mat current_descriptors;
        cv::BFMatcher matcher{detector->defaultNorm()};
        
        //-- Draw keypoints
        //cv::Mat feature_vis;
        //cv::drawKeypoints(current_image, current_keypoints, feature_vis, cv::Scalar{0,255,0});

        std::vector<char> mask;
        std::vector<KeyPoint> moving_keypoints;

        //Only look for matches if we have some features to compare
        if (!previous_descriptors.empty()) {

            std::vector<std::vector<cv::DMatch>> matches;
            std::vector<std::vector<cv::DMatch>> object_matches;

            detector->compute(current_image, current_keypoints, current_descriptors);
            matcher.knnMatch(current_descriptors, previous_descriptors, matches, 2);

            std::vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.5);

            // Only update crosshair position if there is a decent number of matching features
            if (good_matches.size() >= 10) {

                std::vector<cv::Point2f> matching_pts1;
                std::vector<cv::Point2f> matching_pts2;

                // Find matching features
                extract_matching_points(current_keypoints, previous_keypoints,
                                        good_matches, matching_pts1, matching_pts2);

                // Mask features that are not moving
                mask_stationary_features(matching_pts1, matching_pts2, mask);

                // Mask features that do not move the same way as the mean
                mask_false_moving_features(matching_pts2, matching_pts1, mask);

                // Get the moving features
                get_unmasked_keypoints(good_matches, mask, current_keypoints, moving_keypoints);

                // Detected moving object!
                if (moving_keypoints.size() > 10)
                {
                    // Detect keypoint direction


                    // Updating crosshair position to be mean of moving features
                    crosshair_position = calculate_crosshair_position(moving_keypoints);

                    update_object_boundary(object_boundary, moving_keypoints);
                    rectangle_pt1 = cv::Point2d(object_boundary[0], object_boundary[2]); // x1, y1
                    rectangle_pt2 = cv::Point2d(object_boundary[1], object_boundary[3]); // x2, y2

                    // Compute descriptors for the moving features. This can be optimized by looking them up. Currently computes these twice.
                    if(!saved_object) {
                        detector->compute(current_image, moving_keypoints, saved_object_descriptors);
                        saved_object_features = moving_keypoints;
                        current_image.copyTo(object_reference_image);
                        saved_object = true;
                        printf("Object saved!\n");
                        printf("Saved %d features\n", (int)moving_keypoints.size());
                    }
                    else
                    {
                        // Identify new features from object boundary

                    }
                }
                else
                {
                    // DEBUG
                    drawed_mean_vector.x = 0;
                    drawed_mean_vector.y = 0;
                }
            }

            if (saved_object) {
                //look for saved features in the image
                matcher.knnMatch(current_descriptors, saved_object_descriptors, object_matches, 3);

                std::vector<cv::DMatch> good_object_matches = extract_good_ratio_matches(object_matches, 0.7);

                if (good_object_matches.size() > 0) {
                    cv::drawMatches(current_image, current_keypoints, object_reference_image, saved_object_features, good_object_matches, object_vis);
                    //printf("found object!\n");
                }
                //Draw them
                //Update the crosshair position
            }

            // Draw moving features
            cv::drawKeypoints(crosshair_image, moving_keypoints, crosshair_image);

            //cv::drawMatches(current_image, current_keypoints, previous_image, previous_keypoints, good_matches, feature_vis,-1,-1,mask);
            //cv::drawMatches(current_image, current_keypoints, previous_image, previous_keypoints, moving_keypoints, feature_vis);
        }

        // Draw the Crosshair
        cv::drawMarker(crosshair_image, crosshair_position, Scalar::all(255), cv::MARKER_CROSS, 100, 1, 8);
        cv::rectangle(crosshair_image, rectangle_pt1, rectangle_pt2, Scalar::all(255), 1);

        //-- Show detected (drawn) matches
        cv::Mat final_image;

        // DEBUG
        cv::arrowedLine(crosshair_image,
                        cv::Point2d((image_width / RESIZE_FACTOR) / 2, (image_height / RESIZE_FACTOR) / 2),
                        cv::Point2d((image_width / RESIZE_FACTOR) / 2 + drawed_mean_vector.x,
                                    (image_height / RESIZE_FACTOR) / 2 + drawed_mean_vector.y),
                        cv::Scalar(0, 0, 255), 1);


        resize(crosshair_image, final_image, Size(image_width, image_height), 0, 0, INTER_LINEAR);
        //resize(object_vis, final_image, Size(image_width*4, image_height*2), 0, 0, INTER_LINEAR);
        //imshow(matches_win, object_vis);
        imshow(matches_win, final_image);

        previous_image = current_image;
        previous_keypoints = current_keypoints;
        previous_descriptors = current_descriptors;

        int key = cv::waitKey(30);
        if (key == 'q') break;
        if (key == 'r') saved_object = false;
    }
}