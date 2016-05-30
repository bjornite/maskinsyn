//
// Created by mathiact on 5/15/16.
//

#include "Feature_tracker.h"

// Constructor
Feature_tracker::Feature_tracker (
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

    // Initialize feature detector and matcher
    detector = cv::xfeatures2d::SURF::create();
    matcher = {detector->defaultNorm()};

    // Set crosshair to center
    crosshair_position = cv::Point2i(resized_image_size.width/2, resized_image_size.height/2);
    rectangle_center = crosshair_position;
}

vector<cv::DMatch> Feature_tracker::extract_good_ratio_matches (
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
void Feature_tracker::extract_matching_points (
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
float Feature_tracker::euclidian_distance (
        cv::Point2f pt1,
        cv::Point2f pt2)
{
    return sqrt(pow(pt1.x - pt2.x,2) + pow(pt1.y - pt2.y,2));
}

// Calculates the confidence value given crosshair movement and the object size
double Feature_tracker::calculate_confidence_value ()
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
bool Feature_tracker::point_is_within_rectangle (
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
void Feature_tracker::mask_stationary_features (
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
void Feature_tracker::mask_false_moving_features (
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
void Feature_tracker::get_unmasked_keypoints (
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
}

// Extracts true-entries corresponding to the mask
void Feature_tracker::filter_keypoints (
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

// Extracts true-entries corresponding to the mask
void Feature_tracker::filter_descriptors (
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

// Extracts true-entries corresponding to the mask
void Feature_tracker::filter_keypoints_and_descriptors (
        const vector<cv::KeyPoint>& input_keypoints,
        const cv::Mat& input_descriptors,
        const vector<char>& mask,
        vector<cv::KeyPoint>& output_keypoints,
        cv::Mat& output_descriptors)
{
    // Control size
    if (input_keypoints.size() != mask.size() || input_descriptors.rows != mask.size()){
        CV_Error(cv::Error::StsBadSize, "Features and mask must be the same size");
    }

    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            output_keypoints.push_back(input_keypoints.at(i));
            output_descriptors.push_back(input_descriptors.row(i));
        }
    }
}

// TODO DEBUG-function
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
    for (int i = 0; i < mask.size(); i++) {
        if (mask.at(i)) {
            unmasked_descriptors.push_back(descriptors.row(matches.at(i).queryIdx));
        }
    }
}

// Returns good matches
vector<cv::DMatch> Feature_tracker::get_matches (
        const cv::Mat& descriptors_1,
        const cv::Mat& descriptors_2)
{
    vector<vector<cv::DMatch>> object_matches;
    matcher.knnMatch(descriptors_1, saved_object_descriptors, object_matches, 2);
    return extract_good_ratio_matches(object_matches, 0.5);

}

// Extracts all matching keypoints
void Feature_tracker::get_matching_keypoints (
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
void Feature_tracker::add_new_keypoints_to_model (
        const vector<cv::KeyPoint> &keypoints,
        const cv::Mat &descriptors)
{
    for (int i = 0; i < descriptors.rows; i++) {

        // Create descriptor row to be inserted in to the descriptors mat
        cv::Mat descriptor_row = descriptors.row(i);

        // Only save 1000 descriptors
        if (additional_object_descriptors.rows < 1000) {
            additional_object_descriptors.push_back(descriptor_row);
        }

        // Overwrite the correct row with the new descriptor
        else {
            // Reset next_descriptor_to_overwrite if it is out of bounds
            if (next_descriptor_to_overwrite >= additional_object_descriptors.rows) {
                next_descriptor_to_overwrite = 0;
            }
            descriptor_row.copyTo(additional_object_descriptors.row(next_descriptor_to_overwrite++));
        }
    }
}

// Returns keypoints within the rectangle
vector<cv::KeyPoint> Feature_tracker::get_rectangle_keypoints (
        const vector<cv::KeyPoint>& image_keypoints)
{
    vector<cv::KeyPoint> rectangle_keypoints;

    for (int i = 0; i < image_keypoints.size(); i++) {
        cv::KeyPoint keypoint = image_keypoints.at(i);

        if (point_is_within_rectangle(keypoint.pt)) {
            rectangle_keypoints.push_back(keypoint);
        }
    }

    return rectangle_keypoints;
}


// Extract keypoints and their descriptors within the rectangle
void Feature_tracker::get_rectangle_keypoints_and_descriptors (
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
void Feature_tracker::create_mahalanobis_mask (
        const vector<cv::KeyPoint>& keypoints,
        vector<char>& mask)
{
    // We must at least have one keypoint
    if (keypoints.size() > 0) {
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
            mask.push_back(cv::Mahalanobis(samples.row(i), mean, covariance_matrix.inv()) <
                           mahalanobis_threshold_object_model);
        }
    }
}

// Locates new rectangle matches and saves them to the rectangle model
void Feature_tracker::find_and_save_new_rectangle_matches (
        const vector<cv::KeyPoint>& current_keypoints,
        const cv::Mat& current_descriptors)
{
    // Get rectangle keypoints and descriptors
    vector<vector<cv::DMatch>> rectangle_matches;
    vector<cv::KeyPoint> rectangle_keypoints;
    cv::Mat rectangle_descriptors;
    get_rectangle_keypoints_and_descriptors(current_keypoints, current_descriptors,
                                            rectangle_keypoints, rectangle_descriptors);

    // Look for new rectangle matches (based on prev image)
    if (!previous_rectangle_keypoints.empty()) {

        // Get moving keypoints and descriptors
        vector<cv::KeyPoint> moving_rectangle_keypoints;
        cv::Mat moving_rectangle_descriptors;
        get_moving_keypoints_and_descriptors(rectangle_keypoints, rectangle_descriptors, previous_rectangle_keypoints,
                                             previous_rectangle_descriptors, moving_rectangle_keypoints, moving_rectangle_descriptors);

        // Add new keypoints that moved within the rectangle to the new_object_model
        if (!moving_rectangle_keypoints.empty()) {
            add_new_keypoints_to_model(moving_rectangle_keypoints, moving_rectangle_descriptors);
        } else {
            add_new_keypoints_to_model(moving_rectangle_keypoints, moving_rectangle_descriptors);
        }
    }

    // Update previous pointes
    previous_rectangle_keypoints = rectangle_keypoints;
    previous_rectangle_descriptors = rectangle_descriptors;
}

// Updates crosshair position to be the mean of the given keypoints
void Feature_tracker::update_crosshair_position(
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

    crosshair_position = cv::Point2d(mean_x, mean_y);
}

// Updates the object boundary (rectangle) and size
void Feature_tracker::update_object_boundary (
        vector<cv::KeyPoint>& keyPoints)
{
    // Find max, min keypoint in x and y
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

    rectangle_center = crosshair_position;

    // Updating object size
    object_size[0] = maxX - minX;
    object_size[1] = maxY - minY;
}

void Feature_tracker::track (
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

    //Only look for matches if we have some features to compare
    if (!previous_descriptors.empty()) {

        // Try to create model between previous and current image if we have no model
        if (!saved_object) {
            try_to_create_object_model(current_keypoints, current_descriptors, previous_keypoints, previous_descriptors,
                                       current_image);
        }
        else {
            // Look for saved features in the image
            vector<vector<cv::DMatch>> additional_matches;

            // Look for object matches with the original model
            vector<cv::DMatch> good_object_matches = get_matches(current_descriptors, saved_object_descriptors);

            // Extract the matching keypoints
            vector<cv::KeyPoint> matching_keypoints;
            get_matching_keypoints(good_object_matches, current_keypoints, matching_keypoints);

            // Look for additional matches in the additional model (rectangle)
            vector<cv::KeyPoint> additional_matching_keypoints;
            if (!additional_object_descriptors.empty()) {
                matcher.knnMatch(current_descriptors, additional_object_descriptors, additional_matches, 2);
            }
            vector<cv::DMatch> additional_good_matches = extract_good_ratio_matches(additional_matches, 0.5);
            get_matching_keypoints(additional_good_matches, current_keypoints, additional_matching_keypoints);

            // Get the matched keypoints within the rectangle
            additional_matching_keypoints = get_rectangle_keypoints(additional_matching_keypoints);

            // Filter both match results with mahalanobis distance
            vector<char> mask;
            vector<cv::KeyPoint> filtered_keypoints, filtered_additional_keypoints;

            // Model matches
            create_mahalanobis_mask(matching_keypoints, mask);
            filter_keypoints(matching_keypoints, mask, filtered_keypoints);
            mask.clear();

            // Rectangle mathces
            create_mahalanobis_mask(additional_matching_keypoints, mask);
            filter_keypoints(additional_matching_keypoints, mask, filtered_additional_keypoints);

            // Add matches from the additional model to all matches if there are some
            vector<cv::KeyPoint> all_matching_keypoints = filtered_keypoints;
            if (!filtered_additional_keypoints.empty())
                all_matching_keypoints.insert(all_matching_keypoints.end(),
                                              filtered_additional_keypoints.begin(),
                                              filtered_additional_keypoints.end());

            // Find new keypoints in the rectangle and save them to the additional object model
            Feature_tracker::find_and_save_new_rectangle_matches(current_keypoints, current_descriptors);

            // Update crosshair position and confidence_value
            // Set rectangle position to crosshair position if we have 20% of original keypoints
            if (filtered_keypoints.size() >= saved_object_keypoints.size() * 0.2) {
                confidence_value = 0.8 + 0.2 * (good_object_matches.size() / (double) saved_object_descriptors.rows);
                update_crosshair_position(filtered_keypoints);
            }
            else {
                if (all_matching_keypoints.size() > 0) {
                    update_crosshair_position(all_matching_keypoints);

                    calculate_confidence_value();
                } else
                    confidence_value = 0;
            }

            // Draw matched keypoints, model matches green, additional matches red
            cv::drawKeypoints(crosshair_image, filtered_keypoints, crosshair_image, cv::Scalar(0, 255, 0));
            cv::drawKeypoints(crosshair_image, additional_matching_keypoints, crosshair_image, cv::Scalar(0, 255, 255));
        }
    }
    // We do not have previous descriptors, we have no idea what we are tracking
    else {
        confidence_value = 0;
    }

    update_rectangle_position();

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

void Feature_tracker::reset()
{
    printf("Resetting model..\n");

    crosshair_position = cv::Point2i(0,0);
    rectangle_center = cv::Point2i(0,0);
    rectangle_speed[0] = 0;
    rectangle_speed[1] = 0;
    object_image.release();
    confidence_value = 0;
    saved_object = false;
}

// TODO DEBUG/imshow
void Feature_tracker::show_debug_images (
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

// Tries to find a moving object and saves it if it does
void Feature_tracker::try_to_create_object_model (
        const vector<cv::KeyPoint>& current_keypoints,
        const cv::Mat& current_descriptors,
        const vector<cv::KeyPoint>& previous_keypoints,
        const cv::Mat& previous_descriptors,
        const cv::Mat& current_image)
{
    // Get moving keypoints and descriptors
    vector<cv::KeyPoint> moving_keypoints;
    cv::Mat moving_descriptors;
    get_moving_keypoints_and_descriptors(current_keypoints, current_descriptors, previous_keypoints,
                                         previous_descriptors, moving_keypoints, moving_descriptors);

    // Check if we have enough moving points to call it an object
    if (moving_keypoints.size() > minimum_moving_features) {

        // Update crosshair position to be mean of moving features
        update_crosshair_position(moving_keypoints);

        // Update rectangle position
        update_object_boundary(moving_keypoints);

        // Save object image, descriptors and keypoints
        saved_object_keypoints = moving_keypoints;
        saved_object_descriptors = moving_descriptors;
        set_object_image();
        saved_object = true;

        printf("Object saved, with %d features\n", (int) moving_keypoints.size());
    }
}

// Extracts moving keypoints and descriptors
void Feature_tracker::get_moving_keypoints_and_descriptors (
        const vector<cv::KeyPoint>& current_keypoints,
        const cv::Mat& current_descriptors,
        const vector<cv::KeyPoint>& previous_keypoints,
        const cv::Mat& previous_descriptors,
        vector<cv::KeyPoint>& output_keypoints,
        cv::Mat& output_descriptors)
{
    // Find matches between previous and current matches
    vector<vector<cv::DMatch>> matches;
    matcher.knnMatch(current_descriptors, previous_descriptors, matches, 2);
    vector<cv::DMatch> good_matches = extract_good_ratio_matches(matches, 0.5);

    if (good_matches.size() >= minimum_matching_features) {
        vector<cv::Point2f> matching_pts1;
        vector<cv::Point2f> matching_pts2;

        // Get the matching keypoints
        extract_matching_points(current_keypoints, previous_keypoints,
                                good_matches, matching_pts1, matching_pts2);

        // Mask features that are not moving
        vector<char> mask;
        mask_stationary_features(matching_pts1, matching_pts2, mask, 2);

        // Mask features that do not move the same way as the mean
        mask_false_moving_features(matching_pts2, matching_pts1, mask);

        // Get the moving features
        vector<cv::KeyPoint> moving_keypoints;
        cv::Mat moving_descriptors;
        get_unmasked_keypoints(good_matches, mask, current_keypoints, moving_keypoints);
        get_unmasked_descriptors(good_matches, mask, current_descriptors, moving_descriptors);

        // Refine the keypoints further, remove features with high mahalanobis distance
        vector<cv::KeyPoint> refined_moving_keypoints;
        cv::Mat refined_moving_descriptors;

        // Create mask
        vector<char> mahalanobis_mask;
        create_mahalanobis_mask(moving_keypoints, mahalanobis_mask);

        // Filter keypoints and descriptors with it
        filter_keypoints_and_descriptors(moving_keypoints, moving_descriptors,
                                         mahalanobis_mask, output_keypoints, output_descriptors);
    }
}

// Wipes the additional saved keypoints
void Feature_tracker::wipe_rectangle_model ()
{
    additional_object_descriptors.release();
}

// Returns the current confidence value
double Feature_tracker::get_confidence_value ()
{
    return confidence_value;
}


cv::Point2i Feature_tracker::get_object_position ()
{
    cv::Point2d resized_point;
    resized_point.x = rectangle_center.x * resize_factor;
    resized_point.y = rectangle_center.y * resize_factor;

    return resized_point;
}

void Feature_tracker::set_object_image ()
{
    cv::Rect2d rectangle;
    rectangle.x = object_boundary[0];
    rectangle.y = object_boundary[2];

    rectangle.width = object_size[0];
    rectangle.height = object_size[1];

    object_image = crosshair_image(rectangle);

    // Show the image with keypoints
    // Draw keypoints
    cv::Mat object_keypoints_image;
    cv::drawKeypoints(crosshair_image, saved_object_keypoints, object_keypoints_image);

    // Crop and resize
    object_keypoints_image = object_keypoints_image(rectangle);
    cv::resize(object_keypoints_image, object_keypoints_image,
               cv::Size(object_image.cols*resize_factor*2, object_image.rows*resize_factor*2));

    string object_window = "Saved object";
    cv::namedWindow(object_window);
    cv::imshow(object_window, object_keypoints_image);
}

// Crops and extracts the rectangle image and converts it to 64-bit l*a*b
cv::Mat Feature_tracker::get_object_image_lab ()
{
    // Crop it to half the size, extract center
    cv::Rect2d rectangle;
    rectangle.width = object_image.cols/2;
    rectangle.height = object_image.rows/2;

    rectangle.x = object_image.cols/2 - rectangle.width/2;
    rectangle.y = object_image.rows/2 - rectangle.height/2;

    cv::Mat cropped_image = object_image(rectangle);

    // Convert to l*a*b 64
    cv::Mat image_lab, image_lab_64;
    cv::cvtColor(cropped_image, image_lab, CV_BGR2Lab);
    image_lab.convertTo(image_lab_64, CV_64FC3);

    return image_lab_64;
}

void Feature_tracker::tune_pid (char command)
{
    if (command == 'y') PID_kP += 0.1;
    if (command == 'h') PID_kP -= 0.1;
    if (command == 'u') PID_kI += 0.1;
    if (command == 'j') PID_kI -= 0.1;
    if (command == 'i') PID_kD += 0.01;
    if (command == 'k') PID_kD -= 0.01;

    printf("PID: %.2f, %.2f, %.2f\n", PID_kP, PID_kI, PID_kD);

}

// Updates the rectangle position with the movement controller
void Feature_tracker::update_rectangle_position ()
{
    // Output
    float p_out[2];
    float i_out[2];
    float d_out[2];

    // Calculate current error (distance to target)
    float error[2];
    error[0] = crosshair_position.x - rectangle_center.x;
    error[1] = crosshair_position.y - rectangle_center.y;
    p_out[0] = error[0];
    p_out[1] = error[1];

    // Calculate integral part
    PID_integral[0] += error[0]*PID_dT;
    PID_integral[1] += error[1]*PID_dT;
    i_out[0] = PID_integral[0];
    i_out[1] = PID_integral[1];

    // Calculate derivative part
    d_out[0] = (error[0] - PID_previous_error[0])/PID_dT;
    d_out[1] = (error[1] - PID_previous_error[1])/PID_dT;

    // Update previous error
    PID_previous_error[0] = error[0];
    PID_previous_error[1] = error[1];

    // Set speed
    rectangle_speed[0] = p_out[0]*PID_kP + i_out[0]*PID_kI + d_out[0]*PID_kD;
    rectangle_speed[1] = p_out[1]*PID_kP + i_out[1]*PID_kI + d_out[1]*PID_kD;

    rectangle_center.x += rectangle_speed[0];
    rectangle_center.y += rectangle_speed[1];

    rectangle_pt1 = cv::Point2d(rectangle_center.x - object_size[0]/2, rectangle_center.y - object_size[1]/2);
    rectangle_pt2 = cv::Point2d(rectangle_center.x + object_size[0]/2, rectangle_center.y + object_size[1]/2);
}

// Returns true if an object is saved
bool Feature_tracker::found_object ()
{
    return saved_object;
}