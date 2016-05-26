//
// Created by bjornivar on 19.05.16.
//

#ifndef MASKINSYN_Color_model_object_tracker_H
#define MASKINSYN_Color_model_object_tracker_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string.h>

class Color_model_object_tracker {

public:
    // Constructor
    Color_model_object_tracker(double MAX_MAHALANOBIS_DISTANCE, int RESIZE_FACTOR);

    // Standard method, returns a thresholded image using the model. Generates the model if necessary.
    void segment(cv::Mat image, cv::Mat& dst_image);

    // Takes a CV_64FC3 image
    void train(cv::Mat samples);

    void retrain();

    // Takes a CV_64FC3 image
    cv::Mat mahalanobis_distance_for_each_pixel(cv::Mat image_lab_64);

    // Takes a CV_8U mask with all values either 0 or 255
    cv::Mat refineMask(cv::Mat mask);

    void calculateObjectPosition(cv::Mat mask);
    void calculateConfidenceValue(int nrOfPointsWithinModelThreshold);

    // Drawing methods
    void drawInfo(cv::Mat& image);
    void drawMask(cv::Mat image, cv::Mat mask);

    // Methods for getting parameters from the model
    cv::Point2d get_object_position();
    double get_confidence_value();

    // These methods are not used at the moment
    void normalizeL(cv::Mat& image);
    void makeABmatrix(cv::Mat& image_lab, cv::Mat& image_ab);
    cv::Mat make_mahalanobis_image(cv::Mat image_lab_64);
    void otsu(cv::Mat mahalanobis_image, cv::Mat& mask);

    // Methods for setting parameters
    void increaseRefinementIterations();
    void decreaseRefinementIterations();
    void increaseRefinementKernelSize();
    void decreaseRefinementKernelSize();
    void increaseMahalanobisDistance();
    void decreaseMahalanobisDistance();

private:
    cv::Mat mean;
    cv::Mat covariance_matrix;
    cv::Mat covariance_matrix_uchar;
    cv::Mat inv_covariance_matrix;
    double confidenceValue;
    cv::Point2d crossHairPosition;
    double MAX_MAHALANOBIS_DISTANCE;
    int RESIZE_FACTOR;
    int refinement_iterations;
    int refinement_size;
    bool trained;

};

#endif // MASKINSYN_Color_model_object_tracker_H
