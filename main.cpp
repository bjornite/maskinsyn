#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/xfeatures2d.hpp>

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
    std::string keypoints1 = "Undistorted image with keypoints";
    cv::namedWindow(keypoints1);
    std::string keypoints2 = "First image with keypoints";
    cv::namedWindow(keypoints2);

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

    cv::Mat last_image;
    cap >> last_image;

    cv::Mat base_descriptors;

    while(true) {
        cv::Mat raw_image;
        cap >> raw_image;

        cv::Mat current_image; // Will be the undistorted version of the above image.

        cv::undistort(raw_image, current_image, Camera_Matrix, Distortion_Coefficients);

        cv::imshow(undistorted_win, current_image);

        cv::aruco::detectMarkers(current_image, dictionary, markerCorners, markerIds);

        cv::Mat imageCopy;
        current_image.copyTo(imageCopy);

        Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

        std::vector<KeyPoint> current_keypoints;
        detector->detect( current_image, current_keypoints);
        cv::BFMatcher matcher{detector->defaultNorm()};
        
        //-- Draw keypoints
        Mat img_keypoints_1, img_keypoints_2;
        drawKeypoints( current_image, current_keypoints, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        drawKeypoints( last_image, last_keypoints, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

        if (!last_descriptors.empty())
        {
            cv::Mat frame_descriptors;
            std::vector<std::vector<cv::DMatch>> matches;
            desc_extractor->compute(gray_frame, frame_keypoints, frame_descriptors);
            matcher.knnMatch(frame_descriptors, base_descriptors, matches, 2);
            std::vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.8);

            cv::drawMatches(current_image, current_keypoints, last_image, base_keypoints, good_matches, feature_vis);

            if (good_matches.size() >= 10)
            {
                std::vector<cv::Point2f> matching_pts1;
                std::vector<cv::Point2f> matching_pts2;
                extract_matching_points(current_keypoints, last_keypoints,
                                        good_matches, matching_pts1, matching_pts2);

                //// Estimate homography in a ransac scheme
                //cv::Mat is_inlier;
                //find_homography_ransac(matching_pts1, matching_pts2, is_inlier);

                //// Improve homography estimate by normalized DLT
                //cv::Matx33d H = find_homography_normalized_DLT(
                //  sample_Point2f(matching_pts1, is_inlier),
                //  sample_Point2f(matching_pts2, is_inlier));

                cv::Matx33d H = cv::findHomography(matching_pts1, matching_pts2, cv::RANSAC);


                //-- Show detected (drawn) keypoints
        imshow(keypoints1, img_keypoints_1 );
        imshow(keypoints2, img_keypoints_2);

        last_image = current_image;
        last_keypoints;

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
            //printf("%f\n",z);
            double angle = cvFastArctan(x,z);

            //printf("%f\n",angle);

            if(angle > 5 && angle < 180) {
                printf("Kjør til Høyre\n");
            } else if (angle < 355) {
                printf("Kjør til Venstre\n");
            } else if (z > 1) {
                printf("Kjør rett frem\n");
            } else {
                printf("stå stille\n");
            }
        }
        imshow(aruco_win,imageCopy);

        int key = cv::waitKey(30);
        if (key == 'q') break;
    }

}