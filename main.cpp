#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace std;
using namespace cv;

const Matx33d Camera_Matrix{632.29863082751251, 0, 319.5, 0, 632.29863082751251, 239.5, 0, 0, 1};

const Mat Distortion_Coefficients =
        (Mat_<double>(5,1) << 0.070528331223347215, 0.26247385180956367, 0, 0, -1.0640942232949715);



int main() {

    //Set up windows:
    std::string raw_win = "Raw image";
    cv::namedWindow("raw_win");
    std::string undistorted_win = "Undistorted image";
    cv::namedWindow(undistorted_win);
    std::string aruco_win = "Undistorted image with aruco markers";
    cv::namedWindow(aruco_win);

    //Get image from webcam
    cv::VideoCapture cap{1};
    if (!cap.isOpened()) return -1;


    vector<int> markerIds;
    vector< vector<Point2f> > markerCorners, rejectedCandidates;
    //cv::aruco::DetectorParameters parameters;
    const cv::Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    //cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    while(true) {
        cv::Mat raw_image;
        cap >> raw_image;

        cv::Mat imageUndistorted; // Will be the undistorted version of the above image.

        cv::undistort(raw_image, imageUndistorted, Camera_Matrix, Distortion_Coefficients);

        cv::imshow(undistorted_win, imageUndistorted);
        cv::imshow(raw_win, raw_image);


        cv::aruco::detectMarkers(imageUndistorted, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        cv::Mat imageCopy;
        imageUndistorted.copyTo(imageCopy);

        if(markerIds.size() > 0){
            printf("found marker");
        }


        cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);

        imshow(aruco_win,imageCopy);

        int key = cv::waitKey(30);
        if (key > 0) break;
    }

}