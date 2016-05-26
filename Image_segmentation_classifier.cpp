//
// Created by bjornivar on 19.05.16.
//
#include "Image_segmentation_classifier.h"

Image_segmentation_classifier::Image_segmentation_classifier(double MAX_MAHALANOBIS_DISTANCE)  {
    this->MAX_MAHALANOBIS_DISTANCE = MAX_MAHALANOBIS_DISTANCE;
    refinement_iterations = 1;
    refinement_size = 1;
    trained = false;
}

/*
static cv::Ptr<Image_segmentation_classifier> Image_segmentation_classifier::create() {
    return cv::Ptr();
}
 */

void Image_segmentation_classifier::segment(cv::Mat image, cv::Mat& dst_image) {

    cv::resize(image, image, cv::Size(320,240),0,0,cv::INTER_LINEAR);

    //Convert image to LAB color space and 64-bit float
    cv::Mat image_lab, image_lab_64,image_ab;

    cv::cvtColor(image, image_lab, CV_BGR2Lab);

    //makeABmatrix(image_lab,image_ab);
    //image_ab.convertTo(dst_image,CV_Lab2BGR);
    //image_lab.copyTo(dst_image);

    image_lab.convertTo(image_lab_64, CV_64FC3);

    //image_lab_64.convertTo(dst_image,CV_8UC3);

    //Rectangle stuff:
    cv::Mat reference_rectangle;

    //cv::Rect myROI(280, 350, 80, 80);
    cv::Rect myROI(140, 175, 40, 40);

    reference_rectangle = image_lab_64(myROI).clone();

    //Train the model
    if(!trained) {
        train(reference_rectangle);
    }

    if (trained) {

        cv::Mat mask = mahalanobis_distance_for_each_pixel(image_lab_64);
        cv::Mat opened_mask = refineMask(mask);

        calculateObjectPosition(mask);

        drawMask(image_lab,opened_mask);
    }


    //Put some useful information on the output image
    char text1[40];
    char text2[40];
    char text3[40];
    sprintf(text1, "refinement iterations: %d", refinement_iterations);
    sprintf(text2, "refinement size: %d", refinement_size);
    sprintf(text3, "Max M distance: %.3f", MAX_MAHALANOBIS_DISTANCE);
    cv::putText(image_lab, text1, cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, 125, 1, cv::LINE_4);
    cv::putText(image_lab, text2, cv::Point(4, 40), cv::FONT_HERSHEY_SIMPLEX, 0.7, 125, 1, cv::LINE_4);
    cv::putText(image_lab, text3, cv::Point(4, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, 125, 1, cv::LINE_4);

    //Draw the reference rectangle
    cv::rectangle(image_lab, myROI, 2, 0);

    //printf("%.1f,%.1f\n",crossHairPosition.x,crossHairPosition.y);

    cv::drawMarker(image_lab, crossHairPosition, cv::Scalar::all(0), cv::MARKER_CROSS, 100, 1, 8);
    //cv::cvtColor(image_lab,dst_image,CV_Lab2BGR);
    image_lab.convertTo(dst_image,CV_Lab2BGR);
    cv::resize(dst_image, dst_image, cv::Size(640,480),0,0,cv::INTER_LINEAR);
    //image_lab.copyTo(dst_image);

}

void Image_segmentation_classifier::normalizeL(cv::Mat& image) {
    image.forEach<cv::Point3_<double>>([](cv::Point3_<double> &p, const int * position) -> void {
        p.x = sizeof(double) * rand();
    });
}

cv::Point2d Image_segmentation_classifier::get_object_position() {
    cv::Point2d point;
    point.x = crossHairPosition.x*2;
    point.y = crossHairPosition.y*2;
    return point;
}

double Image_segmentation_classifier::get_confidence_value() {
    return confidenceValue;
}

void Image_segmentation_classifier::makeABmatrix(cv::Mat& image_lab, cv::Mat& image_ab) {

    cv::Mat channels[3];

    cv::split(image_lab,channels);

    channels[0]=cv::Mat::ones(image_lab.rows, image_lab.cols, CV_8UC1) * 128;

    cv::merge(channels,3,image_ab);

}

void Image_segmentation_classifier::drawMask(cv::Mat image, cv::Mat mask) {

    image.forEach<cv::Point3_<char>>([this,&mask,&image](cv::Point3_<char> &p, const int * position) -> void {

        cv::Mat p_as_matrix;
        p_as_matrix.push_back(p.x);
        p_as_matrix.push_back(p.y);
        p_as_matrix.push_back(p.z);

        if(mask.at<uchar>(position) == 255) {
            p.y = (char)255;
        };
    });
}

void Image_segmentation_classifier::calculateObjectPosition(cv::Mat mask) {

    int x = 0;
    int y = 0;
    int counter = 0;

    mask.forEach<uchar>([this,&mask,&counter,&x,&y](uchar p,const int position[]) -> void {
        if(p == 255) {
            x += position[1];
            y += position[0];
            counter++;
        };
    });

    if (counter > 0) {
        crossHairPosition.x = x / counter;
        crossHairPosition.y = y / counter;
    }

    //Simple estimation of confidence value. This should be redone properly.

    double sumOfVariances = 0;

    for (int i = 0; i < covariance_matrix.size[0]; i++) {
        sumOfVariances += covariance_matrix.at<double>(i,i);
    }

    confidenceValue = pow(1 - (sumOfVariances / 15000000),2);

    if (counter > mask.size[0]*mask.size[1] / 3) {
        confidenceValue -= 0.3;
    }
    if (confidenceValue < 0) {
        confidenceValue = 0;
    }
}

void Image_segmentation_classifier::train(cv::Mat reference_rectangle) {

    //normalizeL(reference_rectangle);

    //reshape image to a vector of pixels
    cv::Mat samples = reference_rectangle.reshape(1,reference_rectangle.rows*reference_rectangle.cols).t();

    //Create gaussian model of reference rectangle
    cv::calcCovarMatrix(samples,covariance_matrix,mean,CV_COVAR_COLS + CV_COVAR_NORMAL, CV_64FC3);
    cv::invert(covariance_matrix,inv_covariance_matrix);
    trained = true;
}

cv::Mat Image_segmentation_classifier::mahalanobis_distance_for_each_pixel(cv::Mat image_lab_64) {

    cv::Mat mask = cv::Mat::zeros(image_lab_64.size(),CV_8U);

    //Color the pixels that are within the bounds of the model
    image_lab_64.forEach<cv::Point3_<double>>([this,&mask](cv::Point3_<double> &p, const int position[]) -> void {

        cv::Mat p_as_matrix;
        p_as_matrix.push_back(p.x);
        p_as_matrix.push_back(p.y);
        p_as_matrix.push_back(p.z);

        if(cv::Mahalanobis(p_as_matrix,mean,inv_covariance_matrix) < MAX_MAHALANOBIS_DISTANCE) {
            //p.y = 255;
            mask.at<uchar>(position[0],position[1]) = 255;
        };
    });
    return mask;
}

cv::Mat Image_segmentation_classifier::refineMask(cv::Mat mask) {

    cv::Mat opened_mask;

    cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*refinement_size + 1, 2*refinement_size+1 ), cv::Point( refinement_size, refinement_size ) );

    cv::morphologyEx(mask,opened_mask,cv::MORPH_OPEN,element,cv::Point(-1,-1),refinement_iterations,cv::BORDER_CONSTANT);
    cv::morphologyEx(mask,opened_mask,cv::MORPH_CLOSE,element,cv::Point(-1,-1),refinement_iterations,cv::BORDER_CONSTANT);

    //cv::morphologyEx(mask,opened_mask,cv::MORPH_OPEN,100);

    return opened_mask;
}

void Image_segmentation_classifier::increaseCloseIterations() {
    refinement_iterations += 1;
}
void Image_segmentation_classifier::decreaseCloseIterations() {
    if(refinement_iterations > 1) {
        refinement_iterations -= 1;
    }
}

void Image_segmentation_classifier::increaseCloseSize() {
    refinement_size += 1;
}
void Image_segmentation_classifier::decreaseCloseSize() {
    if(refinement_size > 1) {
        refinement_size -= 1;
    }
}

void Image_segmentation_classifier::retrain(){
    trained = false;
}

void Image_segmentation_classifier::increaseMahalanobisDistance() {
    MAX_MAHALANOBIS_DISTANCE += 0.001;
}
void Image_segmentation_classifier::decreaseMahalanobisDistance() {
    if(MAX_MAHALANOBIS_DISTANCE > 0.001) {
        MAX_MAHALANOBIS_DISTANCE -= 0.001;
    }
}