//
// Created by bjornivar on 11.05.16.
//

#ifndef MASKINSYN_HOMOGRAPHY_ESTIMATION_H
#define MASKINSYN_HOMOGRAPHY_ESTIMATION_H

#include "opencv2/opencv.hpp"

std::vector<cv::Point2f> map_Point2f(const std::vector<cv::Point2f>& srcpts, const cv::Matx33d& T);
std::vector<cv::Point2f> sample_Point2f(const std::vector<cv::Point2f>& keypts, const cv::Mat& idx);
cv::Mat random_idx(int total_size, int sample_size);
cv::Mat eval_homography(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Matx33d& H, const float& t, int& num_inlier);
cv::Matx33d find_normalizing_similarity(const std::vector<cv::Point2f>& pts);
cv::Matx33d find_homography_DLT(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);
cv::Matx33d find_homography_normalized_DLT(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);
cv::Matx33d find_homography_ransac(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, cv::Mat& is_inlier);


#endif //MASKINSYN_HOMOGRAPHY_ESTIMATION_H
