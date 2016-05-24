//
// Created by bjornivar on 19.05.16.
//

#ifndef MASKINSYN_IMAGE_SEGMENTATION_CLASSIFIER_H
#define MASKINSYN_IMAGE_SEGMENTATION_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string.h>

class Image_segmentation_classifier {

public:
    Image_segmentation_classifier(double MAX_MAHALANOBIS_DISTANCE);

   // static cv::Ptr<Image_segmentation_classifier> create();
    void segment(cv::Mat image, cv::Mat& dst_image);

    // Takes a CV_64FC3 image
    void train(cv::Mat samples);

    // Takes a CV_64FC3 image
    cv::Mat mahalanobis_distance_for_each_pixel(cv::Mat image_lab_64);

    //Takes a CV_8U mask with all values either 0 or 255
    cv::Mat refineMask(cv::Mat mask);

    void drawMask(cv::Mat image, cv::Mat mask);

    void calculateCrosshairPosition(cv::Mat mask);

    void normalizeL(cv::Mat& image);

    void increaseCloseIterations();
    void decreaseCloseIterations();
    void increaseCloseSize();
    void decreaseCloseSize();
    void increaseMahalanobisDistance();
    void decreaseMahalanobisDistance();
    void retrain();

private:
    cv::Mat mean;
    cv::Mat covariance_matrix;
    cv::Mat inv_covariance_matrix;
    cv::Point2d crossHairPosition;
    double MAX_MAHALANOBIS_DISTANCE;
    int refinement_iterations;
    int refinement_size;
    bool trained;

};


#endif //MASKINSYN_IMAGE_SEGMENTATION_CLASSIFIER_H
