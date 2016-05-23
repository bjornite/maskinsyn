//
// Created by bjornivar on 19.05.16.
//

#ifndef MASKINSYN_IMAGE_SEGMENTATION_CLASSIFIER_H
#define MASKINSYN_IMAGE_SEGMENTATION_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

class Image_segmentation_classifier {

public:
    Image_segmentation_classifier();
   // static cv::Ptr<Image_segmentation_classifier> create();
    void segment(cv::Mat image, cv::Mat& dst_image);
    void train(cv::Mat samples);

private:
    cv::Mat mean;
    cv::Mat covariance_matrix;
    cv::Mat inv_covariance_matrix;
};


#endif //MASKINSYN_IMAGE_SEGMENTATION_CLASSIFIER_H
