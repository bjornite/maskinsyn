//
// Created by bjornivar on 19.05.16.
//
#include "Color_model_object_tracker.h"

Color_model_object_tracker::Color_model_object_tracker(double MAX_MAHALANOBIS_DISTANCE, int RESIZE_FACTOR)  {
    this->MAX_MAHALANOBIS_DISTANCE = MAX_MAHALANOBIS_DISTANCE;
    this->RESIZE_FACTOR = RESIZE_FACTOR;
    refinement_iterations = 1;
    refinement_size = 1;
    trained = false;
}

void Color_model_object_tracker::segment(cv::Mat image, cv::Mat& dst_image) {

    cv::resize(image, image, cv::Size(640/RESIZE_FACTOR,480/RESIZE_FACTOR),0,0,cv::INTER_LINEAR);

    //Convert image to LAB color space and 64-bit float
    cv::Mat image_lab, image_lab_64,image_ab;

    cv::cvtColor(image, image_lab, CV_BGR2Lab);
    //image.copyTo(image_lab);
    //makeABmatrix(image_lab,image_ab);
    //image_ab.convertTo(dst_image,CV_Lab2BGR);
    //image_lab.copyTo(dst_image);

    image_lab.convertTo(image_lab_64, CV_64FC3);

    //image_lab_64.convertTo(dst_image,CV_8UC3);

    //Rectangle stuff:
    cv::Mat reference_rectangle;

    //cv::Rect myROI(280, 350, 80, 80);
    cv::Rect myROI(280/RESIZE_FACTOR, 350/RESIZE_FACTOR, 80/RESIZE_FACTOR, 80/RESIZE_FACTOR);

    reference_rectangle = image_lab_64(myROI).clone();

    //Train the model
    if(!trained) {
        train(reference_rectangle);
    }

    if (trained) {

        //cv::Mat mask;
        cv::Mat mask = mahalanobis_distance_for_each_pixel(image_lab_64);
        //cv::Mat mahalanobis_image = make_mahalanobis_image(image_lab_64);
        //cv::resize(mahalanobis_image,dst_image,cv::Size(640,480));
        //otsu(mahalanobis_image, mask);
        cv::Mat opened_mask = refineMask(mask);

        calculateObjectPosition(mask);

        drawMask(image_lab,opened_mask);
    }

    //Put some useful information on the output image
    drawInfo(image_lab);

    //Draw the reference rectangle
    //cv::rectangle(image_lab, myROI, 2, 0);

    //printf("%.1f,%.1f\n",crossHairPosition.x,crossHairPosition.y);

    cv::drawMarker(image_lab, crossHairPosition, cv::Scalar::all(0), cv::MARKER_CROSS, 100, 1, 8);
    //cv::cvtColor(image_lab,dst_image,CV_Lab2BGR);
    image_lab.convertTo(dst_image,CV_Lab2BGR);
    cv::resize(dst_image, dst_image, cv::Size(640,480),0,0,cv::INTER_LINEAR);
    //image_lab.copyTo(dst_image);

}

//Puts some useful stats on the image
void Color_model_object_tracker::drawInfo(cv::Mat& image) {

    char text1[40];
    char text2[40];
    char text3[40];

    sprintf(text1, "Mahalanobis threshold: %.3f", MAX_MAHALANOBIS_DISTANCE);
    sprintf(text2, "refinement size: %d", refinement_size);
    sprintf(text3, "refinement iterations: %d", refinement_iterations);

    cv::putText(image, text1, cv::Point(4, 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, 125, 1, cv::LINE_4);
    //cv::putText(image, text2, cv::Point(4, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 125, 1, cv::LINE_4);
    //cv::putText(image, text3, cv::Point(4, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, 125, 1, cv::LINE_4);
}

//Randomizes the first channel of an image
void Color_model_object_tracker::normalizeL(cv::Mat& image) {
    image.forEach<cv::Point3_<double>>([](cv::Point3_<double> &p, const int * position) -> void {
        p.x = sizeof(double) * rand();
    });
}

cv::Point2d Color_model_object_tracker::get_object_position() {
    cv::Point2d point;
    point.x = crossHairPosition.x*2;
    point.y = crossHairPosition.y*2;
    return point;
}

double Color_model_object_tracker::get_confidence_value() {
    return confidenceValue;
}

//Removes all variation from the first channel of an image.
void Color_model_object_tracker::makeABmatrix(cv::Mat& image_lab, cv::Mat& image_ab) {

    cv::Mat channels[3];

    cv::split(image_lab,channels);

    channels[0]=cv::Mat::ones(image_lab.rows, image_lab.cols, CV_8UC1) * 128;

    cv::merge(channels,3,image_ab);

}

//draws the mask onto the image by setting the middle channel of the image to 255
//Assumes image is of type CV_8UC3
void Color_model_object_tracker::drawMask(cv::Mat image, cv::Mat mask) {

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

//Calculates the mean position of the "true"(255) pixels in the mask
void Color_model_object_tracker::calculateObjectPosition(cv::Mat mask) {

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
    double sumOfStdDeviations = 0;

    for (int i = 0; i < covariance_matrix.size[0]; i++) {
        sumOfStdDeviations += sqrt(abs(covariance_matrix.at<char>(i,i)));
    }

    // 100 turned out to be a suitable number
    confidenceValue = ((1 - (sumOfStdDeviations / 100)) * (1 - ((double)counter/(double)(mask.size[0]*mask.size[1]))));

    // Hacky way of making sure the confidencevalue is within its bounds
    if (confidenceValue < 0) {
        confidenceValue = 0;
    } else if (confidenceValue > 1) {
        confidenceValue = 1;
    }
    printf("%.5f\n",confidenceValue);
}

//Creates a multivariate gaussian model of the pixel colors in the image
void Color_model_object_tracker::train(cv::Mat reference_rectangle) {

    //normalizeL(reference_rectangle);

    //reshape image to a vector of pixels
    cv::Mat samples = reference_rectangle.reshape(1,reference_rectangle.rows*reference_rectangle.cols).t();

    //Create gaussian model of reference rectangle
    cv::calcCovarMatrix(samples,covariance_matrix,mean,CV_COVAR_COLS + CV_COVAR_NORMAL, CV_64FC3);
    cv::invert(covariance_matrix,inv_covariance_matrix);
    covariance_matrix.convertTo(covariance_matrix_uchar,CV_8S);
    trained = true;
}

//Creates a mask from thresholding on the mahalanobis distance of each pixel
//We set this up as a standalone method because we wanted to use a threshold of type double
cv::Mat Color_model_object_tracker::mahalanobis_distance_for_each_pixel(cv::Mat image_lab_64) {

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

//Creates a cv::Mat with type CV_8U representing the mahalanobis distance for each pixel in the input image
//Lower values in the returned image means the pixel is close to the model
cv::Mat Color_model_object_tracker::make_mahalanobis_image(cv::Mat image_lab_64) {

    cv::Mat mahalanobis_image = cv::Mat::zeros(image_lab_64.size(),CV_8U);

    //Color the pixels that are within the bounds of the model
    image_lab_64.forEach<cv::Point3_<double>>([this,&mahalanobis_image](cv::Point3_<double> &p, const int position[]) -> void {

        cv::Mat p_as_matrix;
        p_as_matrix.push_back(p.x);
        p_as_matrix.push_back(p.y);
        p_as_matrix.push_back(p.z);

        mahalanobis_image.at<char>(position[0],position[1]) = (char)(255 * cv::Mahalanobis(p_as_matrix,mean,inv_covariance_matrix));

    });
    return mahalanobis_image;
}

//Thresholds the image using Otsus method.
void Color_model_object_tracker::otsu(cv::Mat mahalanobis_image, cv::Mat& mask) {
    cv::threshold(mahalanobis_image,mask,cv::THRESH_BINARY,255,cv::THRESH_OTSU);
}

//Refines the mask by opening and then closing to remove noise and give a smoother mask
cv::Mat Color_model_object_tracker::refineMask(cv::Mat mask) {

    cv::Mat opened_mask;

    cv::Mat element = cv::getStructuringElement( 2, cv::Size( 2*refinement_size + 1, 2*refinement_size+1 ), cv::Point( refinement_size, refinement_size ) );

    cv::morphologyEx(mask,opened_mask,cv::MORPH_OPEN,element,cv::Point(-1,-1),refinement_iterations,cv::BORDER_CONSTANT);
    cv::morphologyEx(opened_mask,opened_mask,cv::MORPH_CLOSE,element,cv::Point(-1,-1),refinement_iterations,cv::BORDER_CONSTANT);

    //cv::morphologyEx(mask,opened_mask,cv::MORPH_OPEN,100);

    return opened_mask;
}

//The program will retrain the model the next time segment is called
void Color_model_object_tracker::retrain(){
    trained = false;
}

//These functions all simply alter a parameter of the model
void Color_model_object_tracker::increaseCloseIterations() {
    refinement_iterations += 1;
}

void Color_model_object_tracker::decreaseCloseIterations() {
    if(refinement_iterations > 1) {
        refinement_iterations -= 1;
    }
}

void Color_model_object_tracker::increaseCloseSize() {
    refinement_size += 1;
}

void Color_model_object_tracker::decreaseCloseSize() {
    if(refinement_size > 1) {
        refinement_size -= 1;
    }
}

void Color_model_object_tracker::increaseMahalanobisDistance() {
    MAX_MAHALANOBIS_DISTANCE += 0.001;
}

void Color_model_object_tracker::decreaseMahalanobisDistance() {
    if(MAX_MAHALANOBIS_DISTANCE > 0.001) {
        MAX_MAHALANOBIS_DISTANCE -= 0.001;
}
    }
