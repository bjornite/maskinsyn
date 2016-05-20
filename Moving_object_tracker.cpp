//
// Created by mathiact on 5/15/16.
//

#include "Moving_object_tracker.h"

// Constructor
Moving_object_tracker::Moving_object_tracker (
        int max_keypoints,
        int minimum_matching_features,
        float min_movement_threshold,
        float movement_vector_similarity_threshold,
        int resize_factor
        ) : minimum_matching_features(minimum_matching_features),
                             min_movement_threshold(min_movement_threshold),
                             movement_vector_similarity_threshold(movement_vector_similarity_threshold),
                             resize_factor(resize_factor)
{

    // Calculate resized image size
    resized_image_size = cv::Size(image_width / resize_factor, image_height / resize_factor);

    min_pixel_movement = ((image_width / resize_factor) / 100) * min_movement_threshold;

    // Initialize feature detector
    detector = cv::xfeatures2d::SURF::create(max_keypoints);

    crosshair_position = cv::Point2d(0, 0);

    // Find keypoints and their descriptors for the first image
    detector->detect( previous_image, previous_keypoints);
    detector->compute(previous_image, previous_keypoints, previous_descriptors);

}

vector<cv::DMatch> Moving_object_tracker::extract_good_ratio_matches (
        const vector<vector<cv::DMatch>>& matches,
        double max_ratio)
{
    vector<cv::DMatch> good_ratio_matches;

    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < matches[i][1].distance * max_ratio)
            good_ratio_matches.push_back(matches[i][0]);
    }

    return good_ratio_matches;
}

// Extracts matched points from matches into keypts1 and 2
void Moving_object_tracker::extract_matching_points (
        const vector<cv::KeyPoint>& keypts1,
        const vector<cv::KeyPoint>& keypts2,
        const vector<cv::DMatch>& matches,
        vector<cv::Point2f>& matched_pts1,
        vector<cv::Point2f>& matched_pts2)
{
    matched_pts1.clear();
    matched_pts2.clear();
    for (int i = 0; i < matches.size(); ++i)
    {
        matched_pts1.push_back(keypts1[matches[i].queryIdx].pt);
        matched_pts2.push_back(keypts2[matches[i].trainIdx].pt);
    }
}

// Returns the euclidian distance between two points in 2D-space
float Moving_object_tracker::euclidian_distance (
        cv::Point2f pt1,
        cv::Point2f pt2)
{
    return sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
}

// Fills a binary mask given the distance between matching points
void Moving_object_tracker::mask_stationary_features (
        const vector<cv::Point2f>& matched_pts1,
        const vector<cv::Point2f>& matched_pts2,
        vector<char>& mask)
{
    mask.clear();

    //for_each(matched_pts1.begin(), matched_pts1.end(), [] cv::Point2d pt) {mask.push_back()}

    for (int i = 0; i < matched_pts1.size(); i++) {
        mask.push_back(euclidian_distance(matched_pts1.at(i), matched_pts2.at(i)) > min_pixel_movement);
    }
}

// Updates a binary mask by removing hipsters (points moving differently from the mean)
void Moving_object_tracker::mask_false_moving_features (
        const vector<cv::Point2f>& matched_pts1,
        const vector<cv::Point2f>& matched_pts2,
        vector<char>& mask)
{
    // Find mean vector
    float direction[2] = {0, 0};
    int unmasked_cnt = 0;

    // Only use non-masked keypoints
    for (int i = 0; i < matched_pts1.size(); i++)
    {
        if (mask.at(i) == true) {
            direction[0] += matched_pts2.at(i).x - matched_pts1.at(i).x;
            direction[1] += matched_pts2.at(i).y - matched_pts1.at(i).y;
            unmasked_cnt++;
        }
    }

    direction[0] /= unmasked_cnt;
    direction[1] /= unmasked_cnt;

    cv::Point2d mean_vector(direction[0], direction[1]);
    drawn_mean_vector = mean_vector;

    // DEBUG
    double mean_direction = atan2f(direction[1], direction[0]) * (180 / M_PI);
    //printf("Mean angle: %d\n", (int)mean_direction);

    // Find minimum allowed euclidian distance from the mean vector
    float mean_vector_length = sqrt(pow(direction[0], 2) + pow(direction[1], 2));
    float minimum_distance = mean_vector_length * movement_vector_similarity_threshold;

    // Mask keypoints with euclidian distance from mean greater than MOVEMENT_VECTOR_SIMILARITY_THRESHOLD
    for (int i = 0; i < matched_pts1.size(); i++)
    {
        // Only look at non-masked keypoints
        if (mask.at(i) == true) {
            // Create vector from point1 to point2
            int x = matched_pts2.at(i).x - matched_pts1.at(i).x;
            int y = matched_pts2.at(i).y - matched_pts1.at(i).y;
            cv::Point2d p(x, y);
            float a = euclidian_distance(p, mean_vector);
            float b = minimum_distance;

            // Mask points with greater euclidian distance to mean vector than threshold
            if (euclidian_distance(p, mean_vector) > minimum_distance) {
                mask.at(i) = false;
            }
        }
    }
}

// Masks matches and returns a vector with the matches corresponding to true-entries in the mask
void Moving_object_tracker::get_unmasked_keypoints (
        const vector<cv::DMatch>& matches,
        const vector<char>& mask,
        const vector<cv::KeyPoint>& keypoints,
        vector<cv::KeyPoint>& unmasked_keypoints)
{
    if (matches.size() != mask.size()){
        CV_Error(cv::Error::StsBadSize,"matches and mask must be the same size");
    }
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            unmasked_keypoints.push_back(keypoints.at(matches.at(i).queryIdx));
        }
    }
}

// Updates crosshair position to be the mean of the given keypoints
cv::Point2d Moving_object_tracker::calculate_crosshair_position (
        const vector<cv::KeyPoint> &keypoints)
{
    double mean_x = 0;
    double mean_y = 0;

    for (int i = 0; i < keypoints.size(); i++)
    {
        mean_x += keypoints.at(i).pt.x;
        mean_y += keypoints.at(i).pt.y;
    }

    mean_x /= keypoints.size();
    mean_y /= keypoints.size();

    return cv::Point2d(mean_x, mean_y);
}

// Updates the object boundary (rectangle)
void Moving_object_tracker::update_object_boundary (
        int object_boundary[],
        vector<cv::KeyPoint>& keyPoints)
{
    int minX = keyPoints.at(0).pt.x;
    int maxX = keyPoints.at(0).pt.x;
    int minY = keyPoints.at(0).pt.y;
    int maxY = keyPoints.at(0).pt.y;

    for (int i = 0; i < keyPoints.size(); i++)
    {
        if (keyPoints.at(i).pt.x < minX) minX = keyPoints.at(i).pt.x;
        if (keyPoints.at(i).pt.x > maxX) maxX = keyPoints.at(i).pt.x;
        if (keyPoints.at(i).pt.y < minY) minY = keyPoints.at(i).pt.y;
        if (keyPoints.at(i).pt.y > maxY) maxY = keyPoints.at(i).pt.y;
    }

    object_boundary[0] = minX;
    object_boundary[1] = maxX;
    object_boundary[2] = minY;
    object_boundary[3] = maxY;
}

void Moving_object_tracker::track(
        cv::Mat& input_image,
        cv::Mat& output_image)
{

    // Make image smaller to save computation power
    resize(input_image, current_image, resized_image_size, 0, 0, cv::INTER_LINEAR);

    // Copy image to crosshair image
    current_image.copyTo(crosshair_image);

    // Find keypoints
    vector<cv::KeyPoint> current_keypoints;
    detector->detect(current_image, current_keypoints);


    cv::Mat current_descriptors; // previous descriptors declared in class
    detector->compute(current_image, current_keypoints, current_descriptors);

    cv::BFMatcher matcher{detector->defaultNorm()};

    // Find out which keypoints are moving
    vector<char> mask;
    vector<cv::KeyPoint> moving_keypoints;

    //Only look for matches if we have some features to compare
    if (!previous_descriptors.empty()) {

        vector<vector<cv::DMatch>> matches;
        vector<vector<cv::DMatch>> object_matches;

        matcher.knnMatch(current_descriptors, previous_descriptors, matches, 2);

        vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.5);

        // Only update crosshair position if there is a decent number of matching features
        if (good_matches.size() >= 10) {

            vector<cv::Point2f> matching_pts1;
            vector<cv::Point2f> matching_pts2;

            // Find matching features
            extract_matching_points(current_keypoints, previous_keypoints,
                                    good_matches, matching_pts1, matching_pts2);

            // Mask features that are not moving
            mask_stationary_features(matching_pts1, matching_pts2, mask);

            // Mask features that do not move the same way as the mean
            mask_false_moving_features(matching_pts2, matching_pts1, mask);

            // Get the moving features
            get_unmasked_keypoints(good_matches, mask, current_keypoints, moving_keypoints);

            // Detected moving object!
            if (moving_keypoints.size() > 10)
            {
                // Detect keypoint direction


                // Updating crosshair position to be mean of moving features
                crosshair_position = calculate_crosshair_position(moving_keypoints);

                update_object_boundary(object_boundary, moving_keypoints);
                rectangle_pt1 = cv::Point2d(object_boundary[0], object_boundary[2]); // x1, y1
                rectangle_pt2 = cv::Point2d(object_boundary[1], object_boundary[3]); // x2, y2

                // Compute descriptors for the moving features. This can be optimized by looking them up. Currently computes these twice.
                if(!saved_object) {
                    detector->compute(current_image, moving_keypoints, saved_object_descriptors);
                    saved_object_features = moving_keypoints;
                    current_image.copyTo(object_reference_image);
                    saved_object = true;
                    printf("Object saved!\n");
                    printf("Saved %d features\n", (int)moving_keypoints.size());
                }
                else
                {
                    // Identify new features from object boundary

                }
            }
            else
            {
                // DEBUG
                drawn_mean_vector.x = 0;
                drawn_mean_vector.y = 0;
            }
        }

        if (saved_object) {
            //look for saved features in the image
            matcher.knnMatch(current_descriptors, saved_object_descriptors, object_matches, 3);

            vector<cv::DMatch> good_object_matches = extract_good_ratio_matches(object_matches, 0.7);

            if (good_object_matches.size() > 0) {
                cv::drawMatches(current_image, current_keypoints, object_reference_image, saved_object_features, good_object_matches, object_vis);
                //printf("found object!\n");
            }
            //Draw them
            //Update the crosshair position
        }

        // Draw moving features
        cv::drawKeypoints(crosshair_image, moving_keypoints, crosshair_image);

        //cv::drawMatches(current_image, current_keypoints, previous_image, previous_keypoints, good_matches, feature_vis,-1,-1,mask);
        //cv::drawMatches(current_image, current_keypoints, previous_image, previous_keypoints, moving_keypoints, feature_vis);
    }

    // Draw the Crosshair
    cv::drawMarker(crosshair_image, crosshair_position, cv::Scalar::all(255), cv::MARKER_CROSS, 100, 1, 8);
    cv::rectangle(crosshair_image, rectangle_pt1, rectangle_pt2, cv::Scalar::all(255), 1);

    // DEBUG
    cv::arrowedLine(crosshair_image,
                    cv::Point2d((image_width / resize_factor) / 2, (image_height / resize_factor) / 2),
                    cv::Point2d((image_width / resize_factor) / 2 + drawn_mean_vector.x,
                                (image_height / resize_factor) / 2 + drawn_mean_vector.y),
                    cv::Scalar(0, 0, 255), 1);


    resize(crosshair_image, output_image, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
    //resize(object_vis, final_image, Size(image_width*4, image_height*2), 0, 0, INTER_LINEAR);

    previous_image = current_image;
    previous_keypoints = current_keypoints;
    previous_descriptors = current_descriptors;
}

void Moving_object_tracker::reset()
{
    saved_object = false;
}