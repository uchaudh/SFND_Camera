#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void magnitudeSobel()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1gray.png");

    // convert image to grayscale
    cv::Mat imgGray = img.clone();
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // apply smoothing
    cv::Mat imgblurred = imgGray.clone();
    cv::GaussianBlur(imgGray,imgblurred,cv::Size(5,5),2.0);

    // create filter kernels both for x and y
    float sobel_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1};
    cv::Mat kernel_x = cv::Mat(3,3,CV_32F, sobel_x);

    float sobel_y[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1};
    cv::Mat kernel_y = cv::Mat(3,3,CV_32F, sobel_y);

    // apply filter
    cv::Mat result_x, result_y;
    cv::filter2D(imgblurred, result_x, -1, kernel_x, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(imgblurred, result_y, -1, kernel_y, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

    // compute magnitude image
    cv::Mat magnitude = imgGray.clone();
    for(int r = 0; r < magnitude.rows; r++)
    {
        for(int c = 0; c < magnitude.cols; c++){
            magnitude.at<unsigned char>(r,c) = sqrt(pow(result_x.at<unsigned char>(r,c),2) + 
                                                    pow(result_y.at<unsigned char>(r,c),2));
        }
    }

    // show result
    string windowName = "Magnitude Sobel";
    cv::namedWindow(windowName, 1); // create window
        cv::imshow("Original", img);
    cv::imshow(windowName, magnitude);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    magnitudeSobel();
}