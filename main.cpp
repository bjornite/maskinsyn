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
    //const cv::Ptr<cv::aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    //cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    /*
    std::string marker_win = "Marker image";
    cv::namedWindow("marker_win");
    cv::Mat markerImage;

    cv::aruco::drawMarker(dictionary, 23, 200, markerImage, 1);
    imwrite("image.jpg",markerImage);
    cv::imshow(marker_win, markerImage);
     */

    while(true) {
        cv::Mat raw_image;
        cap >> raw_image;

        cv::Mat imageUndistorted; // Will be the undistorted version of the above image.

        cv::undistort(raw_image, imageUndistorted, Camera_Matrix, Distortion_Coefficients);

        cv::imshow(undistorted_win, imageUndistorted);

        cv::aruco::detectMarkers(imageUndistorted, dictionary, markerCorners, markerIds);

        cv::Mat imageCopy;
        imageUndistorted.copyTo(imageCopy);

        if(markerIds.size() > 0) {

            std::vector<Vec3d> rvecs, tvecs;

            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.1952, Camera_Matrix, Distortion_Coefficients, rvecs,
                                                 tvecs);

            //cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
            cv::aruco::drawAxis(imageCopy, Camera_Matrix, Distortion_Coefficients, rvecs, tvecs, 0.5);

            //Code for controlling the robot goes here:
            //henter x-koordinatet til arucomarkeren i kameraets koordinatsystem
            double x = tvecs[0][0];
            //printf("%f\n",x);
            double z = tvecs[0][2];

            double angle = cvFastArctan(x,z);

            printf("%f\n",angle);

            if(angle > 5 && angle < 180) {
                printf("Kjør til Høyre\n");
            } else if (angle < 355) {
                printf("Kjør til Venstre\n");
            }
        }
        imshow(aruco_win,imageCopy);

        int key = cv::waitKey(30);
        if (key == 'q') break;
    }

}