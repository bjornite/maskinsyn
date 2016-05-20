//
// Created by bjornivar on 19.05.16.
//

#include "Image_segmentation_classifier.h"

Image_segmentation_classifier::Image_segmentation_classifier(){
}

/*
static cv::Ptr<Image_segmentation_classifier> Image_segmentation_classifier::create() {
    return cv::Ptr();
}
 */

void Image_segmentation_classifier::segment(cv::Mat image, cv::Mat & dst_image) {

    //Convert image to LAB color space and 64-bit float
    cv::Mat image_lab,image_lab_64;

    cv::cvtColor(image,image_lab,CV_BGR2Lab);
    image_lab.convertTo(image_lab_64,CV_64FC3);


    //Rectangle stuff:
    cv::Mat reference_rectangle;

    cv::Rect myROI(280, 350, 80, 80);
    cv::Point2d seedPoint(320,240);

    cv::rectangle(image_lab_64,myROI,2,0);

    reference_rectangle = image_lab(myROI).clone();

    //Create samples
    cv::Mat samples = reference_rectangle.reshape(1, reference_rectangle.rows*reference_rectangle.cols).t();
    cv::Mat samples_64;
    samples.convertTo(samples_64,CV_64FC3);

    //Create gaussian model of reference rectangle
    cv::calcCovarMatrix(samples_64,covariance_matrix,mean,CV_COVAR_COLS + CV_COVAR_NORMAL, CV_64FC3);
    cv::invert(covariance_matrix,inv_covariance_matrix);

    //Set the mahalanobis threshold
    double mahalanobis_threshold = 0.2;

    //Color the pixels that are within the bounds of the model
    image_lab_64.forEach<cv::Point3_<double>>([this,mahalanobis_threshold](cv::Point3_<double> &p, const int * position) -> void {

        cv::Mat p_as_matrix;
        p_as_matrix.push_back(p.x);
        p_as_matrix.push_back(p.y);
        p_as_matrix.push_back(p.z);

        if(cv::Mahalanobis(p_as_matrix,mean,inv_covariance_matrix) < mahalanobis_threshold) {
            p.y = 255;
        };
    });

    image_lab_64.convertTo(dst_image,CV_8UC3);
}

void Image_segmentation_classifier::train(cv::Mat samples) {

}