#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "Homography_estimation.h"

using namespace cv;
using namespace std;

const Matx33d Camera_Matrix{632.29863082751251, 0, 319.5, 0, 632.29863082751251, 239.5, 0, 0, 1};

const Mat Distortion_Coefficients =
        (Mat_<double>(5,1) << 0.070528331223347215, 0.26247385180956367, 0, 0, -1.0640942232949715);


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
        mask.push_back(euclidian_distance(matched_pts1.at(i), matched_pts2.at(i)) > 5.0);
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

    // Set frame rate to 5
    cap.set(CV_CAP_PROP_FPS, 5);

    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    // Initialiserer variabler for forrige bilde og fyller dem med noe fornuftig
    cv::Mat last_image;
    std::vector<KeyPoint> last_keypoints;
    cv::Mat last_descriptors;

    cap >> last_image;
    detector->detect( last_image, last_keypoints);
    detector->compute(last_image, last_keypoints, last_descriptors);

    while(true) {
        cv::Mat raw_image;
        cv::Mat raw_image_grayscale;
        cap >> raw_image;
        //cv::cvtColor(raw_image,raw_image_grayscale,cv::COLOR_BGR2GRAY);

        cv::Mat current_image; // Will be the undistorted version of the above image.

        cv::undistort(raw_image, current_image, Camera_Matrix, Distortion_Coefficients);

        cv::Mat imageCopy;
        current_image.copyTo(imageCopy);


        std::vector<KeyPoint> current_keypoints;
        detector->detect( current_image, current_keypoints);

        cv::Mat current_descriptors;

        cv::BFMatcher matcher{detector->defaultNorm()};
        
        //-- Draw keypoints
        cv::Mat feature_vis;

        //cv::drawKeypoints(current_image, current_keypoints, feature_vis, cv::Scalar{0,255,0});
        std::vector<char> mask;
        std::vector<KeyPoint> moving_features;

        if (!last_descriptors.empty()) {

            std::vector<std::vector<cv::DMatch>> matches;

            detector->compute(current_image, current_keypoints, current_descriptors);
            matcher.knnMatch(current_descriptors, last_descriptors, matches, 2);

            std::vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.7);


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

                //// Estimate homography in a ransac scheme
                //cv::Mat is_inlier;
                //find_homography_ransac(matching_pts1, matching_pts2, is_inlier);

                //// Improve homography estimate by normalized DLT
                //cv::Matx33d H = find_homography_normalized_DLT(
                //  sample_Point2f(matching_pts1, is_inlier),
                //  sample_Point2f(matching_pts2, is_inlier));

                //cv::Matx33d H = cv::findHomography(matching_pts1, matching_pts2, cv::RANSAC);
            }
            //cv::drawMatches(current_image, current_keypoints, last_image, last_keypoints, good_matches, feature_vis,-1,-1,mask);
            //cv::drawMatches(current_image, current_keypoints, last_image, last_keypoints, moving_features, feature_vis);

            cv::drawKeypoints(current_image, moving_features, feature_vis);
        }

        //-- Show detected (drawn) matches
        imshow(matches_win, feature_vis);

        last_image = current_image;
        last_keypoints = current_keypoints;
        last_descriptors = current_descriptors;

        int key = cv::waitKey(30);
        if (key == 'q') break;
    }

}