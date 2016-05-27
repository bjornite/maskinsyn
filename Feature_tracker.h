//
// Created by mathiact on 5/15/16.
//

#ifndef MASKINSYN_MOVING_OBJECT_DETECTOR_H
#define MASKINSYN_MOVING_OBJECT_DETECTOR_H

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

class Feature_tracker {
    // Higher value = less pixels (faster)
    int resize_factor;

    // This factor multiplied with the mean movement vector length gives the euclidian distance threshold
    // for features to count as part of the tracked object
    // Lower means stricter, a good value is between 0.2 and 0.5
    float movement_vector_similarity_threshold;

    const int image_width = 640;
    const int image_height = 480;

    cv::Size resized_image_size;

    int minimum_matching_features;
    int minimum_moving_features;

    float mahalanobis_threshold_object_model = 0.25;
    float mahalanobis_threshold_rectangle = 0.2;

    // How confident we are in tracking of the right object
    double confidence_value = 0;

    // x1, x2, y1, y2
    int object_boundary[4];

    // Previous image pointer
    cv::Mat previous_image;

    // Cropped image of the object
    cv::Mat object_image;

    // Detector and matcher
    cv::Ptr<cv::xfeatures2d::SURF> detector;
    cv::BFMatcher matcher;

    // Keypoints and descriptors (previous)
    vector<cv::KeyPoint> previous_keypoints;
    cv::Mat previous_descriptors;

    // Set up crosshair image
    cv::Mat crosshair_image;
    cv::Point2i crosshair_position;
    cv::Point2i previous_crosshair_position;

    // Set up rectangle
    cv::Point2d rectangle_pt1, rectangle_pt2;

    // Rectangle component speed
    cv::Point2f rectangle_center;
    int rectangle_speed[2];

    // Previous rectangle keypoints and descriptors
    vector<cv::KeyPoint> previous_rectangle_keypoints;
    cv::Mat previous_rectangle_descriptors;

    // The model of the object
    bool saved_object = false;
    vector<cv::KeyPoint> saved_object_keypoints;
    cv::Mat saved_object_descriptors;

    // Additional descriptors
    int next_descriptor_to_overwrite = 0;
    cv::Mat additional_object_descriptors;

    // Object size, x, y in pixels
    int object_size[2];

public:
    Feature_tracker (
            int max_keypoints = 400,
            int minimum_matching_features = 10,
            int minimum_moving_features = 10,
            float movement_vector_similarity_threshold = 0.3,
            int resize_factor = 1);

    vector<cv::DMatch> extract_good_ratio_matches (
            const vector<vector<cv::DMatch>>& matches,
            double max_ratio);

    void extract_matching_points (
            const vector<cv::KeyPoint>& keypts1,
            const vector<cv::KeyPoint>& keypts2,
            const vector<cv::DMatch>& matches,
            vector<cv::Point2f>& matched_pts1,
            vector<cv::Point2f>& matched_pts2);

    float euclidian_distance (
            cv::Point2f pt1,
            cv::Point2f pt2);

    // Calculates the confidence value given the crosshairs movement
    double calculate_confidence_value ();

    void create_mahalanobis_mask (
            const vector<cv::KeyPoint>& keypoints,
            vector<char>& mask);

    void filter_keypoints (
            const vector<cv::KeyPoint>& keypoints,
            const vector<char>& mask,
            vector<cv::KeyPoint>& filtered_keypoints);

    void filter_descriptors (
            const cv::Mat& descriptors,
            const vector<char>& mask,
            cv::Mat& filtered_descriptors);

    void filter_keypoints_and_descriptors (
            const vector<cv::KeyPoint>& input_keypoints,
            const cv::Mat& input_descriptors,
            const vector<char>& mask,
            vector<cv::KeyPoint>& output_keypoints,
            cv::Mat& output_descriptors);

    // Locates new rectangle matches and saves them to the rectangle model
    void find_and_save_new_rectangle_matches (
            const vector<cv::KeyPoint> &current_keypoints,
            const cv::Mat &current_descriptors);

    bool point_is_within_rectangle (
            cv::Point2f);

    void mask_stationary_features (
            const vector<cv::Point2f> &matched_pts1,
            const vector<cv::Point2f> &matched_pts2,
            vector<char> &mask,
            float min_pixel_movement_percentage);

    void mask_false_moving_features (
            const vector<cv::Point2f>& matched_pts1,
            const vector<cv::Point2f>& matched_pts2,
            vector<char>& mask);

    void get_unmasked_keypoints (
            const vector<cv::DMatch>& matches,
            const vector<char>& mask,
            const vector<cv::KeyPoint>& keypoints,
            vector<cv::KeyPoint>& unmasked_keypoints);

    void get_matching_keypoints (
            const vector<cv::DMatch>& matches,
            const vector<cv::KeyPoint>& keypoints,
            vector<cv::KeyPoint>& matching_keypoints);

    void add_new_keypoints_to_model (
            const vector<cv::KeyPoint>& keypoints,
            const cv::Mat &descriptors);

    void get_rectangle_keypoints_and_descriptors (
            const vector<cv::KeyPoint>& image_keypoints,
            const cv::Mat& image_descriptors,
            vector<cv::KeyPoint>& rectangle_keypoints,
            cv::Mat& rectangle_descriptors);

    void update_crosshair_position(
            const vector<cv::KeyPoint> &keypoints);

    void update_object_boundary (
            int object_boundary[],
            vector<cv::KeyPoint>& keyPoints);

    void track (
            cv::Mat& inputImage,
            cv::Mat& featureImage,
            cv::Mat& outputImage,
            cv::Mat& outputImage2);

    // Tries to find a moving object and saves it if it does
    void try_to_create_object_model (
            const vector<cv::KeyPoint>& current_keypoints,
            const cv::Mat& current_descriptors,
            const vector<cv::KeyPoint>& previous_keypoints,
            const cv::Mat& previous_descriptors,
            const cv::Mat& current_image);

    // Extracts moving keypoints and descriptors
    void get_moving_keypoints_and_descriptors (
            const vector<cv::KeyPoint>& current_keypoints,
            const cv::Mat& current_descriptors,
            const vector<cv::KeyPoint>& previous_keypoints,
            const cv::Mat& previous_descriptors,
            vector<cv::KeyPoint>& output_keypoints,
            cv::Mat& output_descriptors);

    // Updates the rectangle position with the movement controller
    void update_rectangle_position ();

    // Resets the object model
    void reset ();

    // TODO DEBUG/imshow
    void show_debug_images (
            const vector<cv::KeyPoint>& current_keypoints,
            const vector<cv::KeyPoint>& mk,
            const vector<cv::KeyPoint>& moving_keypoints,
            const vector<cv::KeyPoint>& refined_moving_keypoints,
            const cv::Mat& current_image,
            const vector<char>& mask,
            const vector<cv::DMatch>& matches);


    // Wipes the additional saved keypoints
    void wipe_rectangle_model ();

    double get_confidence_value ();

    cv::Point2i get_object_position ();

    void set_object_image();

    // Extracts the rectangle image and converts it to 64-bit l*a*b
    cv::Mat get_object_image_lab();

    // Returns true if an object is saved
    bool found_object ();
};


#endif //MASKINSYN_MOVING_OBJECT_DETECTOR_H
