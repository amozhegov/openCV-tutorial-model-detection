#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"

// CREATE YOUR FUNCTIONS HERE
cv::Mat rva_compute_homography(std::vector<cv::Point2f> points_image1, std::vector<cv::Point2f> points_image2) {
    // Check if the input vectors have exactly 4 points
    if (points_image1.size() != 4 || points_image2.size() != 4) {
        std::cerr << "Error: Both input point vectors must have exactly 4 points." << std::endl;
        exit(1);
    }

    // Compute homography using OpenCV's findHomography function
    cv::Mat homography = cv::findHomography(points_image1, points_image2);

    // Return the resulting homography matrix
    return homography;
}

void rva_draw_contour(cv::Mat image, std::vector<cv::Point2f> points, cv::Scalar color, int thickness) {
    // Create a vector of points to represent the polygon enclosing the marked region
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(std::vector<cv::Point>(points.begin(), points.end()));

    // Draw the polygon on the input image
    cv::drawContours(image, contours, 0, color, thickness);

    // Display the image with the polygon drawn on it
    cv::imshow("Input Image", image);
}

void rva_deform_image(const cv::Mat& im_input, cv::Mat& im_output, cv::Mat homography)
{
    // Get the transformed corners
    std::vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f(0, 0);
    corners[1] = cv::Point2f(im_input.cols, 0);
    corners[2] = cv::Point2f(im_input.cols, im_input.rows);
    corners[3] = cv::Point2f(0, im_input.rows);
    std::vector<cv::Point2f> corners_transformed(4);
    cv::perspectiveTransform(corners, corners_transformed, homography);

    // Debugging: print out the corners and transformed corners
    std::cout << "Corners: " << corners << std::endl;
    std::cout << "Transformed corners: " << corners_transformed << std::endl;

    // Find the bounding box
    cv::Rect bbox = cv::boundingRect(corners_transformed);

    // Debugging: print out the bounding box
    std::cout << "Bounding box: " << bbox << std::endl;

    // Warp the input image
    cv::warpPerspective(im_input, im_output, homography, im_output.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    // Debugging: print out the output image size
    std::cout << "Output image size: " << im_output.size() << std::endl;

//    im_output = im_output(bbox);
    // Show the output image
    cv::imshow("Output Image", im_output);
}
void rva_calcKPsDesc(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    // Initialize SIFT detector and extractor
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    detector->setNFeatures(1000); // set maximum number of keypoints to 1000

    // Detect keypoints and compute descriptors
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

void rva_matchDesc(cv::Mat& descriptors1, cv::Mat& descriptors2, std::vector<cv::DMatch>& matches)
{
    // Initialize descriptor matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");

    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter matches based on distance ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            matches.push_back(knn_matches[i][0]);
        }
    }
}

void rva_drawMatches(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches, cv::Mat& img_matches)
{
    // Initialize output image
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);

    // Show the output image
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);
}
void rva_locateObj(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches, cv::Mat &homography, std::vector<cv::Point2f> &pts_im2)
{
    // Get the matching keypoints from both images
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& match : matches) {
        pts1.push_back(keypoints1[match.queryIdx].pt);
        pts2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Find the homography using RANSAC
    homography = cv::findHomography(pts1, pts2, cv::RANSAC);

    // Calculate the transformed corners of the object in img2
    std::vector<cv::Point2f> corners(4);
    corners[0] = cv::Point2f(0, 0);
    corners[1] = cv::Point2f(img1.cols, 0);
    corners[2] = cv::Point2f(img1.cols, img1.rows);
    corners[3] = cv::Point2f(0, img1.rows);

    cv::perspectiveTransform(corners, pts_im2, homography);
}
