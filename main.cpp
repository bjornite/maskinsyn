#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const Matx33d Camera_Matrix{632.29863082751251, 0, 319.5, 0, 632.29863082751251, 239.5, 0, 0, 1};

const Mat Distortion_Coefficients = (Mat_<double>(5,1) << 0.070528331223347215, 0.26247385180956367, 0, 0, -1.0640942232949715);



int main() {

    //Set up windows:
    std::string raw_win = "Raw image";
    cv::namedWindow("raw_win");
    std::string undistorted_win = "Undistorted image";
    cv::namedWindow(undistorted_win);

    cv::VideoCapture cap{1};
    if (!cap.isOpened()) return -1;


    while(true) {
        cv::Mat image;
        cap >> image;

        cv::Mat imageUndistorted; // Will be the undistorted version of the above image.

        cv::undistort(image, imageUndistorted, Camera_Matrix, Distortion_Coefficients);


        cv::imshow(undistorted_win, imageUndistorted);
        cv::imshow(raw_win, image);

        int key = cv::waitKey(30);
        if (key > 0) break;
    }

}