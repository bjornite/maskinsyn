//
// Created by mathiact on 5/15/16.
//

#ifndef MASKINSYN_MOVING_OBJECT_DETECTOR_H
#define MASKINSYN_MOVING_OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>

class Moving_object_detector
{
    const int image_width = 640;
    const int image_height = 480;

    // The distance features must move per 1/FRAMERATE'th second to count as
    // movement. Measured in percentage of the whole frame size
    float movement_treshold_percentage;

    // Smallest number of matching features required
    int minimum_matching_features;
    float resize_factor;

public:
    Moving_object_detector (float, int, float);

    void draw (cv::Mat&, cv::Mat&, cv::Mat&);
};


#endif //MASKINSYN_MOVING_OBJECT_DETECTOR_H
