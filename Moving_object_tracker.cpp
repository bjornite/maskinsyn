//
// Created by mathiact on 5/15/16.
//

#include "Moving_object_tracker.h"

// Constructor
Moving_object_tracker::Moving_object_tracker (
        int max_keypoints,
        int minimum_matching_features,
        int minimum_moving_features,
        float movement_vector_similarity_threshold,
        int resize_factor) : minimum_matching_features(minimum_matching_features),
                             minimum_moving_features(minimum_moving_features),
                             movement_vector_similarity_threshold(movement_vector_similarity_threshold),
                             resize_factor(resize_factor)
{

    // Calculate resized image size
    resized_image_size = cv::Size(image_width / resize_factor, image_height / resize_factor);

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
    for (int i = 0; i < matches.size(); ++i) {
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

// Returns true if the point is within the object boundary rectangle
bool Moving_object_tracker::point_is_within_rectangle (
        cv::Point2f point)
{
    return point.x > object_boundary[0] &&
           point.x < object_boundary[1] &&
           point.y > object_boundary[2] &&
           point.y < object_boundary[3];
}

// Fills a binary mask given the distance between matching points
// min_pixel_movement_percentage is the distance a feature must move per 1/FRAMERATE second
// to be counted as a moving feature. Givei in percentage of the whole frame width.

void Moving_object_tracker::mask_stationary_features (
        const vector<cv::Point2f>& matched_pts1,
        const vector<cv::Point2f>& matched_pts2,
        vector<char>& mask,
        const int min_pixel_movement_percentage)
{
    mask.clear();

    float min_pixel_movement = ((image_width / resize_factor) / 100) * min_pixel_movement_percentage;

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
    int cnt = 0;
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            cnt++;
            unmasked_keypoints.push_back(keypoints.at(matches.at(i).queryIdx));
        }
    }
    //printf("Found %d/%d moving rectangle keypoints ", cnt, mask.size());
}

// Masks matches and returns a vector with the matches corresponding to true-entries in the mask
void get_unmasked_descriptors (
        const vector<cv::DMatch>& matches,
        const vector<char>& mask,
        const cv::Mat& descriptors,
        cv::Mat& unmasked_descriptors)
{
    if (matches.size() != mask.size()){
        CV_Error(cv::Error::StsBadSize,"matches and mask must be the same size");
    }
    int cnt = 0;
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            cnt++;
            unmasked_descriptors.push_back(descriptors.row(matches.at(i).queryIdx));
        }
    }
}

// Gets all matching keypoints
void Moving_object_tracker::get_matching_keypoints (
        const vector<cv::DMatch>& matches,
        const vector<cv::KeyPoint>& keypoints,
        vector<cv::KeyPoint>& matching_keypoints)
{
    for (int i = 0; i < matches.size(); i++) {
        matching_keypoints.push_back(keypoints.at(matches.at(i).queryIdx));
    }
}

// Adds new keypoints within the rectangle to the model
// (Overwrites old)
void Moving_object_tracker::add_new_keypoints_to_model (
        const vector<cv::KeyPoint> &keypoints,
        const cv::Mat &descriptors)
{
    for (int i = 0; i < descriptors.rows; i++) {

        // Create descriptor row to be inserted in to the descriptors mat
        cv::Mat descriptor_row = descriptors.row(i);

        // Only save 1000 descriptors
        if (new_object_descriptors.rows < 1000) {
            new_object_descriptors.push_back(descriptor_row);
        }

        // Overwrite the correct row with the new descriptor
        else {
            // Reset next_descriptor_to_overwrite if it is out of bounds
            if (next_descriptor_to_overwrite >= new_object_descriptors.rows) {
                next_descriptor_to_overwrite = 0;
            }

            descriptor_row.copyTo(new_object_descriptors.row(next_descriptor_to_overwrite++));
        }
    }
}

// Extract keypoints and their descriptors within the rectangle
void Moving_object_tracker::get_rectangle_keypoints_and_descriptors (
        const vector<cv::KeyPoint>& image_keypoints,
        const cv::Mat& image_descriptors,
        vector<cv::KeyPoint>& rectangle_keypoints,
        cv::Mat& rectangle_descriptors)
{
    for (int i = 0; i < image_keypoints.size(); i++) {
        cv::KeyPoint keypoint = image_keypoints.at(i);
        cv::Mat descriptor = image_descriptors.row(i);

        if (point_is_within_rectangle(keypoint.pt)) {
            rectangle_keypoints.push_back(keypoint);
            rectangle_descriptors.push_back(descriptor);
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

// Updates the object boundary (rectangle) and size
void Moving_object_tracker::update_object_boundary (
        int object_boundary[],
        vector<cv::KeyPoint>& keyPoints)
{
    int minX = (int)keyPoints.at(0).pt.x;
    int maxX = (int)keyPoints.at(0).pt.x;
    int minY = (int)keyPoints.at(0).pt.y;
    int maxY = (int)keyPoints.at(0).pt.y;

    for_each(keyPoints.begin(), keyPoints.end(), [&minX, &maxX, &minY, &maxY](cv::KeyPoint keypoint) mutable {
        if (keypoint.pt.x < minX) minX = (int)keypoint.pt.x;
        if (keypoint.pt.x > maxX) maxX = (int)keypoint.pt.x;
        if (keypoint.pt.y < minY) minY = (int)keypoint.pt.y;
        if (keypoint.pt.y > maxY) maxY = (int)keypoint.pt.y;
    });

    object_boundary[0] = minX;
    object_boundary[1] = maxX;
    object_boundary[2] = minY;
    object_boundary[3] = maxY;

    rectangle_pt1 = cv::Point2d(object_boundary[0], object_boundary[2]); // x1, y1
    rectangle_pt2 = cv::Point2d(object_boundary[1], object_boundary[3]); // x2, y2

    // Updating object size
    object_size[0] = maxX - minX;
    object_size[1] = maxY - minY;
}

void Moving_object_tracker::switch_mode()
{
    if (mode < 1)
        mode++;
    else mode = 0;
}

void Moving_object_tracker::track(
        cv::Mat& input_image,
        cv::Mat& feature_image,
        cv::Mat& output_image,
        cv::Mat& output_image2)
{
    cv::Mat current_image;

    // Make image smaller to save computation power
    resize(input_image, current_image, resized_image_size, 0, 0, cv::INTER_LINEAR);

    // This image will show all current keypoints/features
    current_image.copyTo(feature_image);

    // Copy image to crosshair image
    current_image.copyTo(crosshair_image);

    // DEBUG
    cv::Mat image2;
    current_image.copyTo(image2);

    // TODO Maybe only detect keypoints within the rectangle if it is defined

    // Find keypoints
    vector<cv::KeyPoint> current_keypoints;
    detector->detect(current_image, current_keypoints);

    // Calculate descriptors
    cv::Mat current_descriptors;
    detector->compute(current_image, current_keypoints, current_descriptors);

    cv::BFMatcher matcher{detector->defaultNorm()};
    vector<char> mask;

    //Only look for matches if we have some features to compare
    if (!previous_descriptors.empty()) {

        // Only look at moving features between previous and current image if
        // we have no model
        if (!saved_object) {
            vector<vector<cv::DMatch>> matches;

            matcher.knnMatch(current_descriptors, previous_descriptors, matches, 2);
            vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.5);

            // Only update crosshair position if there is a decent number of matching features
            if (good_matches.size() >= minimum_matching_features) {

                vector<cv::Point2f> matching_pts1;
                vector<cv::Point2f> matching_pts2;

                // Get the matching keypoints
                extract_matching_points(current_keypoints, previous_keypoints,
                                        good_matches, matching_pts1, matching_pts2);

                // Mask features that are not moving
                mask_stationary_features(matching_pts1, matching_pts2, mask, 2);

                // Mask features that do not move the same way as the mean
                mask_false_moving_features(matching_pts2, matching_pts1, mask);

                // Get the moving features
                // List of the keypoints which moved since last image
                vector<cv::KeyPoint> moving_keypoints;
                get_unmasked_keypoints(good_matches, mask, current_keypoints, moving_keypoints);

                // Detected moving object!
                if (moving_keypoints.size() > minimum_moving_features) {
                    // Updating crosshair position to be mean of moving features
                    crosshair_position = calculate_crosshair_position(moving_keypoints);

                    // Only wipe model if we have some moving keypoints
                    if (moving_keypoints.size() > minimum_moving_features) {

                        // Compute descriptors for the moving features. This can be optimized by looking them up.
                        // Currently computes these twice.
                        detector->compute(current_image, moving_keypoints, saved_object_descriptors);
                        saved_object_descriptors.copyTo(new_object_descriptors);

                        saved_object = true;
                        printf("Object saved!\n");
                        printf("Saved %d features\n", (int) moving_keypoints.size());

                        previous_rectangle_descriptors = current_descriptors;
                        previous_rectangle_keypoints = moving_keypoints;

                        saved_object = true;

                        // Update rectangle position
                        update_object_boundary(object_boundary, moving_keypoints);
                        rectangle_pt1 = cv::Point2d(object_boundary[0], object_boundary[2]); // x1, y1
                        rectangle_pt2 = cv::Point2d(object_boundary[1], object_boundary[3]); // x2, y2
                    }
                }
            }
        }
        else
        {
            // Look for saved features in the image
            vector<vector<cv::DMatch>> object_matches, additional_matches, rectangle_matches;

            vector<cv::KeyPoint> rectangle_keypoints;
            cv::Mat rectangle_descriptors;

            get_rectangle_keypoints_and_descriptors(current_keypoints, current_descriptors,
                                                    rectangle_keypoints, rectangle_descriptors);

            matcher.knnMatch(current_descriptors, saved_object_descriptors, object_matches, 2);
            if (!previous_rectangle_keypoints.empty())
                matcher.knnMatch(rectangle_descriptors, previous_rectangle_descriptors, rectangle_matches, 2);
            if (!new_object_descriptors.empty())
                matcher.knnMatch(current_descriptors, new_object_descriptors, additional_matches, 2);


            vector<cv::DMatch> good_object_matches = extract_good_ratio_matches(object_matches, 0.5);
            vector<cv::DMatch> good_rectangle_matches = extract_good_ratio_matches(rectangle_matches, 0.5);
            vector<cv::DMatch> additional_good_matches = extract_good_ratio_matches(additional_matches, 0.5);




                // Get the matching keypoints
                vector<cv::KeyPoint> matching_keypoints, rectangle_matching_keypoints, rectangle_moving_keypoints, additional_matching_keypoints;
                get_matching_keypoints(good_object_matches, current_keypoints, matching_keypoints);
                get_matching_keypoints(good_rectangle_matches, rectangle_keypoints, rectangle_matching_keypoints);
                get_matching_keypoints(additional_good_matches, current_keypoints, additional_matching_keypoints);

                // Get rectangle keypoints that are moving
                vector<char> mask;
                vector<cv::Point2f> rect_points1; // Previous
                vector<cv::Point2f> rect_points2; // Current
                cv::Mat moving_rectangle_descriptors;

                extract_matching_points(rectangle_keypoints, previous_rectangle_keypoints, good_rectangle_matches, rect_points2, rect_points1);

                // Find moving points within rectangle
                mask_stationary_features(rect_points1, rect_points2, mask, 0.5);
                mask_false_moving_features(rect_points1, rect_points2, mask);
                get_unmasked_keypoints(good_rectangle_matches, mask, rectangle_keypoints, rectangle_moving_keypoints);
                get_unmasked_descriptors(good_rectangle_matches, mask, rectangle_descriptors, moving_rectangle_descriptors);

                // Add new keypoints that moved within the rectangle to the new_object_model
                add_new_keypoints_to_model(rectangle_moving_keypoints, moving_rectangle_descriptors);

                vector<cv::KeyPoint> all_matching_keypoints = matching_keypoints;
                all_matching_keypoints.insert(all_matching_keypoints.end(), additional_matching_keypoints.begin(), additional_matching_keypoints.end());


                // Update crosshair position
                //crosshair_position = calculate_crosshair_position(matching_keypoints);
                crosshair_position = calculate_crosshair_position(all_matching_keypoints);
                rectangle_pt1 = cv::Point2d(crosshair_position.x - object_size[0]/2, crosshair_position.y - object_size[1]/2);
                rectangle_pt2 = cv::Point2d(crosshair_position.x + object_size[0]/2, crosshair_position.y + object_size[1]/2);



                // Update rectangle position
                //update_object_boundary(object_boundary, all_matching_keypoints);

                // Draw matched keypoints
                cv::drawKeypoints(crosshair_image, matching_keypoints, crosshair_image);
                cv::drawKeypoints(image2, additional_matching_keypoints, image2);

                // Draw movement arrow
                if (rectangle_moving_keypoints.size() > 5) {
                    cv::arrowedLine(crosshair_image,
                                    crosshair_position,
                                    cv::Point2d((image_width / resize_factor) / 2 + drawn_mean_vector.x,
                                                (image_height / resize_factor) / 2 + drawn_mean_vector.y),
                                    cv::Scalar(0, 0, 255), 1);
                }

                printf("Normal matches: %d, Additional matches: %d, overwrote %d\n", (int)good_object_matches.size(), new_object_descriptors.rows, next_descriptor_to_overwrite);


            // Update previous pointes
            previous_rectangle_keypoints = rectangle_keypoints;
            previous_rectangle_descriptors = rectangle_descriptors;
        }
    }

    // Draw crosshair and rectangle
    cv::drawMarker(crosshair_image, crosshair_position, cv::Scalar::all(255), cv::MARKER_CROSS, 100, 1, 8);
    cv::rectangle(crosshair_image, rectangle_pt1, rectangle_pt2, cv::Scalar::all(255), 1);
    cv::rectangle(image2, rectangle_pt1, rectangle_pt2, cv::Scalar::all(255), 1);

    // Draw all current keypoints to feature image
    cv::drawKeypoints(feature_image, current_keypoints, feature_image);

    // Resize images back to normal
    resize(crosshair_image, output_image, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);
    resize(image2, output_image2, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);
    resize(feature_image, feature_image, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);

    // Update previous pointers
    previous_image = current_image;
    previous_keypoints = current_keypoints;
    previous_descriptors = current_descriptors;
}

void Moving_object_tracker::reset()
{
    printf("Resetting model..\n");
    saved_object = false;
}
