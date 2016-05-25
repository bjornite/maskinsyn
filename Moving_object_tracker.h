//
// Created by mathiact on 5/15/16.
//

#ifndef MASKINSYN_MOVING_OBJECT_DETECTOR_H
#define MASKINSYN_MOVING_OBJECT_DETECTOR_H

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

class Moving_object_tracker {
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

    cv::Point2d drawn_mean_vector;

    // Previous image pointer
    cv::Mat previous_image;

    // Detector and descriptors
    cv::Ptr<cv::xfeatures2d::SURF> detector;
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
    cv::Mat saved_object_descriptors;

    // Additional descriptors
    int next_descriptor_to_overwrite = 0;
    cv::Mat new_object_descriptors;

    // Object size, x, y in pixels
    int object_size[2];

public:
    Moving_object_tracker (
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

    // Removes keypoints not within mahalanobis distance
    void refine_keypoints_mahalanobis (
            const vector<cv::KeyPoint>& keypoints,
            vector<cv::KeyPoint>& output_keypoints);

    cv::Point2d calculate_crosshair_position (
            const vector<cv::KeyPoint> &keypoints);

    void update_object_boundary (
            int object_boundary[],
            vector<cv::KeyPoint>& keyPoints);

    void track (
            cv::Mat& inputImage,
            cv::Mat& featureImage,
            cv::Mat& outputImage,
            cv::Mat& outputImage2);

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
};


#endif //MASKINSYN_MOVING_OBJECT_DETECTOR_H
