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

    //Convert image to LAB color space
    cv::Mat image_lab,reference_rectangle,reference_rectangle_64;
    //cv::cvtColor(image,image_lab,CV_BGR2GRAY);

    cv::cvtColor(image,image_lab,CV_BGR2Lab);

    //Do segmentation
    cv::Rect myROI(280, 350, 80, 80);
    cv::Point2d seedPoint(320,240);

    cv::rectangle(image_lab,myROI,2,0);

    reference_rectangle = image(myROI).clone();

    cv::Mat samples = reference_rectangle.reshape(1, reference_rectangle.rows*reference_rectangle.cols).t();

    cv::Mat samples_64;
    samples.convertTo(samples_64,CV_64FC3);
    //Create gaussian model of reference rectangle

    cv::calcCovarMatrix(samples_64,covariance_matrix,mean,CV_COVAR_COLS + CV_COVAR_NORMAL, CV_64FC3);

    cv::invert(covariance_matrix,inv_covariance_matrix);

    double mahalanobis_threshold = 0.3;
/*
    std::for_each(image_lab.begin(),image_lab.end(),[&image_lab](uchar* &tmp) {
        if(cv::Mahalanobis(tmp,mean,inv_covariance_matrix) > mahalanobis_threshold) {

        };
    });
*/
    //  typedef cv::Vec3b Pixel;

    image_lab.forEach<cv::Point3_<uint8_t>>([this,mahalanobis_threshold](cv::Point3_<uint8_t> &p, const int * position) -> void {

        cv::Mat p_as_matrix;
        p_as_matrix.push_back(p.x);
        p_as_matrix.push_back(p.y);
        p_as_matrix.push_back(p.z);

        cv::Mat p_as_matrix_64;
        p_as_matrix.convertTo(p_as_matrix_64,CV_64F);
        cv::Mat mean_64;
        mean.convertTo(mean_64,CV_64F);

        if(cv::Mahalanobis(p_as_matrix_64,mean_64,inv_covariance_matrix) < mahalanobis_threshold) {
            p.y = 255;
        };
    });

/*
    for (int i = 0; i < image_lab.size().width; i++) {
        for (int j = 0; j < image_lab.size().height; j++) {

            cv::Mat pixel(3,1,CV_8U);
            pixel = image_lab.at<uchar>(i,j);
            cv::Mat pixel_64,mean_pixel_64;
            pixel.convertTo(pixel_64,CV_64F);
            mean.convertTo(mean_pixel_64,CV_64F);

            double p = cv::Mahalanobis(pixel_64, mean_pixel_64, inv_covariance_matrix);
            if (p > mahalanobis_threshold) {
                //Color the pixels
            }
        }
    }
    */
    //Threshold with respect to mahalanobis distance
    //cv::floodFill(image_lab,seedPoint,255,0,2,2);

    //Return image with segmented area in green
    image_lab.copyTo(dst_image);
}

void Image_segmentation_classifier::train(cv::Mat samples) {

}