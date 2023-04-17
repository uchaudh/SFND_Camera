#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    
    vector<cv::KeyPoint> keypoints;
    double max_overlap = 0.0;
    // getting local maxima
    // parse through the image rows
    for(size_t i = 0; i < dst_norm.rows; i++){
        // parse through the image coloumns
        for(size_t j = 0; j < dst_norm.cols; j++){
            // get one pixel
            int response = (int)dst_norm.at<float>(i,j);
            if(response > minResponse){
                cv::KeyPoint new_keypoint;
                new_keypoint.pt = cv::Point2f(j,i);
                new_keypoint.size = apertureSize * 2;
                new_keypoint.response = response;

                // NMS
                bool overlap_flag = false;
                for(auto it = keypoints.begin(); it != keypoints.end(); ++it){
                    double overlap = cv::KeyPoint::overlap(new_keypoint, *it);
                    if(overlap > max_overlap){
                        overlap_flag = true;
                        if(new_keypoint.response > (*it).response){
                            *it = new_keypoint;
                            break;
                        }
                    }
                }
                if(!overlap_flag){
                    keypoints.push_back(new_keypoint);
                }
            }
        }
    }

    // visualize results
    cv::Mat keypointImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, keypointImage, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Harris Corner Detector Response Matrix", dst_norm_scaled);
    cv::imshow(" Keypoints ", keypointImage);
    cv::waitKey(0);

}

int main()
{
    cornernessHarris();
}