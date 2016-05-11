#include <iostream>
#include <opencv2/opencv.hpp>
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
    std::string keypoints1 = "Undistorted image with keypoints";
    cv::namedWindow(keypoints1);
    std::string keypoints2 = "First image with keypoints";
    cv::namedWindow(keypoints2);

    //Get image from webcam
    cv::VideoCapture cap{1};
    if (!cap.isOpened()) return -1;


    Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(400);

    //initialiserer variabler for forrige bilde og fyller dem med noe fornuftig
    cv::Mat last_image;
    std::vector<KeyPoint> last_keypoints;

    cap >> last_image;
    detector->detect( last_image, last_keypoints);

    cv::Mat base_descriptors;

    while(true) {
        cv::Mat raw_image;
        cap >> raw_image;

        cv::Mat current_image; // Will be the undistorted version of the above image.

        cv::undistort(raw_image, current_image, Camera_Matrix, Distortion_Coefficients);

        cv::imshow(undistorted_win, current_image);

        cv::Mat imageCopy;
        current_image.copyTo(imageCopy);


        std::vector<KeyPoint> current_keypoints;
        detector->detect( current_image, current_keypoints);
        cv::BFMatcher matcher{detector->defaultNorm()};
        
        //-- Draw keypoints
        Mat img_keypoints_1, img_keypoints_2;
        drawKeypoints( current_image, current_keypoints, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        drawKeypoints( last_image, last_keypoints, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

        if (!last_descriptors.empty()) {
            cv::Mat frame_descriptors;
            std::vector<std::vector<cv::DMatch>> matches;
            desc_extractor->compute(gray_frame, frame_keypoints, frame_descriptors);
            matcher.knnMatch(frame_descriptors, base_descriptors, matches, 2);
            std::vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.8);

            cv::drawMatches(current_image, current_keypoints, last_image, base_keypoints, good_matches, feature_vis);

            if (good_matches.size() >= 10) {
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
            }
        }


                //-- Show detected (drawn) keypoints
        imshow(keypoints1, img_keypoints_1 );
        imshow(keypoints2, img_keypoints_2);

        last_image = current_image;
        last_keypoints = current_keypoints;

        int key = cv::waitKey(30);
        if (key == 'q') break;
    }

}