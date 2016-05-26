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
    detector = cv::xfeatures2d::SURF::create();

    crosshair_position = cv::Point2i(resized_image_size.width, resized_image_size.height);
    rectangle_center = crosshair_position;

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

// Calculates the confidence value given crosshair movement and the object size
double Moving_object_tracker::calculate_confidence_value ()
{
    double image_diagonal = sqrt(pow((double)resized_image_size.width, 2) + pow((double)resized_image_size.height, 2));
    double crosshair_movement = euclidian_distance(crosshair_position, previous_crosshair_position);
    double crosshair_movement_object_size_factor = crosshair_movement / max(object_size[0], object_size[1]);

    if (crosshair_movement_object_size_factor > 1)
        confidence_value = 0;
    else
        confidence_value = pow((1 - crosshair_movement / image_diagonal), 4) - crosshair_movement_object_size_factor;

    if (confidence_value < 0) confidence_value = 0;
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
// to be counted as a moving feature. Given in percentage of the whole frame width.

void Moving_object_tracker::mask_stationary_features (
        const vector<cv::Point2f>& matched_pts1,
        const vector<cv::Point2f>& matched_pts2,
        vector<char>& mask,
        float min_pixel_movement_percentage)
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
            int x = (int) (matched_pts2.at(i).x - matched_pts1.at(i).x);
            int y = (int) (matched_pts2.at(i).y - matched_pts1.at(i).y);
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

void Moving_object_tracker::filter_keypoints (
        const vector<cv::KeyPoint>& keypoints,
        const vector<char>& mask,
        vector<cv::KeyPoint>& filtered_keypoints)
{
    if (keypoints.size() != mask.size()){
        CV_Error(cv::Error::StsBadSize,"matches and mask must be the same size");
    }
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            filtered_keypoints.push_back(keypoints.at(i));
        }
    }
}

void Moving_object_tracker::filter_descriptors (
        const cv::Mat& descriptors,
        const vector<char>& mask,
        cv::Mat& filtered_descriptors)
{
    if (descriptors.rows != mask.size()){
        CV_Error(cv::Error::StsBadSize,"matches and mask must be the same size");
    }
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            filtered_descriptors.push_back(descriptors.row(i));
        }
    }
}

// TODO DEBUG
void get_unmasked_matches (
        const vector<cv::DMatch>& matches,
        const vector<char>& mask,
        vector<cv::DMatch>& unmasked_matches) {
    if (matches.size() != mask.size()) {
        CV_Error(cv::Error::StsBadSize, "matches and mask must be the same size");
    }
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            unmasked_matches.push_back(matches.at(i));
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

// Creates vector mask, each index is marked true if the corresponding keypoint is within the mahalanobis distance
void Moving_object_tracker::create_mahalanobis_mask (
        const vector<cv::KeyPoint>& keypoints,
        vector<char>& mask)
{
    cv::Mat samples;
    cv::Mat covariance_matrix;
    cv::Mat mean;

    // Create sample matrix, one keypoint, x and y on each row
    for (int i = 0; i < keypoints.size(); i++) {
        cv::Mat row;
        row.push_back(keypoints.at(i).pt.x);
        row.push_back(keypoints.at(i).pt.y);

        samples.push_back(row.t());
    }

    cv::calcCovarMatrix(samples, covariance_matrix, mean, CV_COVAR_ROWS + CV_COVAR_NORMAL, CV_32F);

    // Set mask indexes
    for (int i = 0; i < keypoints.size(); i++) {
        mask.push_back(cv::Mahalanobis(samples.row(i), mean, covariance_matrix.inv()) < mahalanobis_threshold_object_model);
    }
}

// Removes keypoints not within mahalanobis distance
void Moving_object_tracker::refine_keypoints_mahalanobis (
        const vector<cv::KeyPoint>& keypoints,
        vector<cv::KeyPoint>& output_keypoints)
{
    vector<cv::KeyPoint> ditched_moving_keypoints;
    cv::Mat samples;
    cv::Mat covariance_matrix;
    cv::Mat mean;

    // Create sample matrix, one keypoint, x and y on each row
    for (int i = 0; i < keypoints.size(); i++) {
        cv::Mat row;
        row.push_back(keypoints.at(i).pt.x);
        row.push_back(keypoints.at(i).pt.y);

        samples.push_back(row.t());
    }

    cv::calcCovarMatrix(samples, covariance_matrix, mean, CV_COVAR_ROWS + CV_COVAR_NORMAL, CV_32F);

    // Extract features within mahalanobis distance
    for (int i = 0; i < keypoints.size(); i++) {
        if(cv::Mahalanobis(samples.row(i), mean, covariance_matrix.inv()) < mahalanobis_threshold_object_model)
            output_keypoints.push_back(keypoints.at(i));
        else
            ditched_moving_keypoints.push_back(keypoints.at(i));
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

    // TODO DEBUG/imshow
    cv::Mat image2;
    current_image.copyTo(image2);

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

                // TODO DEBUG/imshow
                vector<cv::KeyPoint> mk;
                get_unmasked_keypoints(good_matches, mask, current_keypoints, mk);

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

                        // Refine the keypoints, remove keypoints with high mahalanobis distance
                        vector<cv::KeyPoint> refined_moving_keypoints;
                        refine_keypoints_mahalanobis(moving_keypoints, refined_moving_keypoints);

                        show_debug_images(current_keypoints, mk, moving_keypoints, refined_moving_keypoints,
                                          current_image, mask, good_matches);

                        // Compute descriptors for the moving features. This can be optimized by looking them up.
                        // Currently computes these twice.
                        detector->compute(current_image, refined_moving_keypoints, saved_object_descriptors);

                        saved_object = true;
                        printf("Object saved!\n");
                        printf("Saved %d features\n", (int) moving_keypoints.size());

                        previous_rectangle_descriptors = current_descriptors;
                        previous_rectangle_keypoints = moving_keypoints;

                        // Update rectangle position
                        update_object_boundary(object_boundary, refined_moving_keypoints);
                        rectangle_pt1 = cv::Point2d(object_boundary[0], object_boundary[2]); // x1, y1
                        rectangle_pt2 = cv::Point2d(object_boundary[1], object_boundary[3]); // x2, y2
                        rectangle_center = crosshair_position;

                        // Save object image
                        set_object_image();

                        saved_object = true;
                    }
                }
            }
        }
        else
        {
            // Look for saved features in the image
            vector<vector<cv::DMatch>> object_matches, additional_matches, rectangle_matches;

            // Look for object matches
            matcher.knnMatch(current_descriptors, saved_object_descriptors, object_matches, 2);
            vector<cv::DMatch> good_object_matches = extract_good_ratio_matches(object_matches, 0.5);


            // RECTANGLE *************************************************************************************
            // Get rectangle keypoints and descriptors
            vector<cv::KeyPoint> rectangle_keypoints;
            cv::Mat rectangle_descriptors;
            get_rectangle_keypoints_and_descriptors(current_keypoints, current_descriptors,
                                                    rectangle_keypoints, rectangle_descriptors);

            // Look for new rectangle matches (based on prev image)
            if (!previous_rectangle_keypoints.empty()) {
                matcher.knnMatch(rectangle_descriptors, previous_rectangle_descriptors, rectangle_matches, 2);
            }
            vector<cv::DMatch> good_rectangle_matches = extract_good_ratio_matches(rectangle_matches, 0.5);

            // Look for additional matches in the rectangle model
            vector<cv::KeyPoint> additional_matching_keypoints;
            if (!new_object_descriptors.empty()) {
                matcher.knnMatch(current_descriptors, new_object_descriptors, additional_matches, 2);
            }
            vector<cv::DMatch> additional_good_matches = extract_good_ratio_matches(additional_matches, 0.5);
            get_matching_keypoints(additional_good_matches, current_keypoints, additional_matching_keypoints);

            // Get the matching keypoints
            vector<cv::KeyPoint> matching_keypoints, rectangle_matching_keypoints, rectangle_moving_keypoints;
            get_matching_keypoints(good_object_matches, current_keypoints, matching_keypoints);
            get_matching_keypoints(good_rectangle_matches, rectangle_keypoints, rectangle_matching_keypoints);


            // Get rectangle keypoints that are moving
            vector<char> mask;
            vector<cv::Point2f> rect_points1; // Previous
            vector<cv::Point2f> rect_points2; // Current
            cv::Mat moving_rectangle_descriptors;

            extract_matching_points(rectangle_keypoints, previous_rectangle_keypoints, good_rectangle_matches, rect_points2, rect_points1);

            mask_stationary_features(rect_points1, rect_points2, mask, 0.5);
            mask_false_moving_features(rect_points1, rect_points2, mask);
            get_unmasked_keypoints(good_rectangle_matches, mask, rectangle_keypoints, rectangle_moving_keypoints);
            get_unmasked_descriptors(good_rectangle_matches, mask, rectangle_descriptors, moving_rectangle_descriptors);

            // Refine rectangle matches by removing features far from mean
            vector<cv::KeyPoint> refined_rectangle_moving_keypoints;
            cv::Mat refined_rectangle_moving_descriptors;
            if (!rectangle_moving_keypoints.empty()) {
                vector<char> mahalanobis_mask;
                create_mahalanobis_mask(rectangle_moving_keypoints, mahalanobis_mask);

                // Filtering out features far from the mean
                filter_keypoints(rectangle_moving_keypoints, mahalanobis_mask, refined_rectangle_moving_keypoints);
                filter_descriptors(moving_rectangle_descriptors, mahalanobis_mask,
                                   refined_rectangle_moving_descriptors);

            }

            // Add new keypoints that moved within the rectangle to the new_object_model
            if (!refined_rectangle_moving_keypoints.empty()) {
                add_new_keypoints_to_model(refined_rectangle_moving_keypoints, refined_rectangle_moving_descriptors);
            } else {
                add_new_keypoints_to_model(rectangle_moving_keypoints, moving_rectangle_descriptors);
            }

            // Add matches from the rectangle model to all matches if there are some
            vector<cv::KeyPoint> all_matching_keypoints = matching_keypoints;
            if (! additional_matching_keypoints.empty())
                all_matching_keypoints.insert(all_matching_keypoints.end(), additional_matching_keypoints.begin(), additional_matching_keypoints.end());

            // Update crosshair position and confidence_value
            // Set rectangle position to crosshair position if we have 20% of original keypoints
            if (good_object_matches.size() >= saved_object_descriptors.rows*0.2) {
                confidence_value = 0.8 + 0.2*(good_object_matches.size() / (double)saved_object_descriptors.rows);
                crosshair_position = calculate_crosshair_position(matching_keypoints);

                rectangle_center.x = crosshair_position.x;
                rectangle_center.y = crosshair_position.y;
                rectangle_speed[0] = 0;
                rectangle_speed[1] = 0;
            } else {
                if (all_matching_keypoints.size() > 0) {
                    crosshair_position = calculate_crosshair_position(all_matching_keypoints);
                    rectangle_center.x = crosshair_position.x;
                    rectangle_center.y = crosshair_position.y;

                    calculate_confidence_value();
                } else
                    confidence_value = 0;
            }

            // Update rectangle position
            //update_object_boundary(object_boundary, all_matching_keypoints);



            // Draw matched keypoints
            cv::drawKeypoints(crosshair_image, matching_keypoints, crosshair_image);
            cv::drawKeypoints(image2, additional_matching_keypoints, image2);

            // Draw movement arrow
            if (rectangle_moving_keypoints.size() > 5) {
                cv::arrowedLine(crosshair_image,
                                cv::Point2d((image_width / resize_factor) / 2,
                                            (image_height / resize_factor) / 2),
                                cv::Point2d((image_width / resize_factor) / 2 + drawn_mean_vector.x,
                                            (image_height / resize_factor) / 2 + drawn_mean_vector.y),
                                cv::Scalar(0, 0, 255), 1);
            }

            //printf("Normal matches: %d, Additional matches: %d\n", (int)good_object_matches.size(), new_object_descriptors.rows);

            // Update previous pointes
            previous_rectangle_keypoints = rectangle_keypoints;
            previous_rectangle_descriptors = rectangle_descriptors;
        }
    }
    // We do not have previous descriptors
    else {
        confidence_value = 0;
    }

    // Update rectangle position
    float x_dist_to_target = (float) (crosshair_position.x - rectangle_center.x);
    float y_dist_to_target = (float) (crosshair_position.y - rectangle_center.y);

    rectangle_speed[0] += x_dist_to_target/5;
    rectangle_speed[1] += y_dist_to_target/5;

    //rectangle_center.x += rectangle_speed[0];
    //rectangle_center.y += rectangle_speed[1];
    rectangle_center.x += x_dist_to_target/3;
    rectangle_center.y += y_dist_to_target/3;

    //printf("Feature confidence: %.2f ", confidence_value);

    rectangle_pt1 = cv::Point2d(rectangle_center.x - object_size[0]/2, rectangle_center.y - object_size[1]/2);
    rectangle_pt2 = cv::Point2d(rectangle_center.x + object_size[0]/2, rectangle_center.y + object_size[1]/2);

    // Reduce rectangle speed due to drag
    rectangle_speed[0] *= 0.99;
    rectangle_speed[1] *= 0.99;

    // Draw crosshair and rectangle
    cv::drawMarker(crosshair_image, crosshair_position, cv::Scalar::all(255), cv::MARKER_CROSS, 100, 1, 8);
    cv::rectangle(crosshair_image, rectangle_pt1, rectangle_pt2, cv::Scalar::all(255), 1);
    cv::rectangle(image2, rectangle_pt1, rectangle_pt2, cv::Scalar::all(255), 1);

    // Draw all current keypoints to feature image
    cv::drawKeypoints(feature_image, current_keypoints, feature_image);

    // Resize images back to normal
    resize(crosshair_image, output_image, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
    resize(image2, output_image2, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
    resize(feature_image, feature_image, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);

    // Update previous pointers
    previous_image = current_image;
    previous_keypoints = current_keypoints;
    previous_descriptors = current_descriptors;
    previous_crosshair_position = crosshair_position;
}

void Moving_object_tracker::reset()
{
    printf("Resetting model..\n");

    crosshair_position = cv::Point2i(0,0);
    rectangle_center = cv::Point2i(0,0);
    rectangle_speed[0] = 0;
    rectangle_speed[1] = 0;
    object_image.release();

    saved_object = false;
}

// TODO DEBUG/imshow
void Moving_object_tracker::show_debug_images (
        const vector<cv::KeyPoint>& current_keypoints,
        const vector<cv::KeyPoint>& mk,
        const vector<cv::KeyPoint>& moving_keypoints,
        const vector<cv::KeyPoint>& refined_moving_keypoints,
        const cv::Mat& current_image,
        const vector<char>& mask,
        const vector<cv::DMatch>& matches)
{
    vector<cv::DMatch> moving_matches;
    get_unmasked_matches(matches, mask, moving_matches);

    std::string af = "All features";
    std::string mf = "Moving features";
    std::string mfsd = "Moving in same direction features";
    std::string rmf = "Refined features";
    std::string vis = "Matches";
    cv::namedWindow(af);
    cv::namedWindow(mf);
    cv::namedWindow(mfsd);
    cv::namedWindow(rmf);
    cv::namedWindow(vis);
    cv::Mat tmp1, tmp2, tmp3, tmp4, tmp5;
    cv::drawKeypoints(crosshair_image, current_keypoints, tmp1);
    cv::drawKeypoints(crosshair_image, mk, tmp2);
    cv::drawKeypoints(crosshair_image, moving_keypoints, tmp3);
    cv::drawKeypoints(crosshair_image, refined_moving_keypoints, tmp4);
    cv::drawMatches(current_image, current_keypoints, previous_image, previous_keypoints, moving_matches, tmp5);
    resize(tmp1, tmp1, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);
    resize(tmp2, tmp2, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);
    resize(tmp3, tmp3, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);
    resize(tmp4, tmp4, cv::Size(image_width*2, image_height*2), 0, 0, cv::INTER_LINEAR);
    resize(tmp5, tmp5, cv::Size(image_width*4, image_height*2), 0, 0, cv::INTER_LINEAR);
    imshow(af, tmp1);
    imshow(mf, tmp2);
    imshow(mfsd, tmp3);
    imshow(rmf, tmp4);
    imshow(vis, tmp5);
}

// Wipes the additional saved keypoints
void Moving_object_tracker::wipe_rectangle_model ()
{
    new_object_descriptors.release();
}

// Returns the current confidence value
double Moving_object_tracker::get_confidence_value ()
{
    return confidence_value;
}


cv::Point2i Moving_object_tracker::get_object_position ()
{
    cv::Point2d resized_point;
    resized_point.x = rectangle_center.x * resize_factor;
    resized_point.y = rectangle_center.y * resize_factor;

    return resized_point;
}

void Moving_object_tracker::set_object_image ()
{
    cv::Rect2d rectangle;
    rectangle.x = object_boundary[0];
    rectangle.y = object_boundary[2];

    rectangle.width = object_size[0];
    rectangle.height = object_size[1];

    object_image = crosshair_image(rectangle);
}

cv::Mat Moving_object_tracker::get_object_image_lab ()
{
    cv::Mat image_lab, image_lab_64;

    cv::cvtColor(object_image, image_lab, CV_BGR2Lab);
    image_lab.convertTo(image_lab_64, CV_64FC3);

    return image_lab_64;
}

bool Moving_object_tracker::found_object ()
{
    return saved_object;
}