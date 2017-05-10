/*
    Reads Images and Detect ORB keypoints
    
    Author : Manohar Kuse <mpkuse@connect.ust.hk>
    Released in Public Domain
*/

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;

int main()
{
  //
  // Load Images
  cv::Mat im = cv::imread( "../image/church1.jpg");
  cv::imshow( "win", im );
  cv::waitKey(0);


  //
  // Feature Detector - ORB
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  fdetector->detectAndCompute(im, cv::Mat(), keypoints, descriptors);
  cout << "# of keypoints : "<< keypoints.size() << endl;
  cout << "descriptors shape : "<< descriptors.rows << "x" << descriptors.cols << endl;


  //
  // Draw Keypoints
  cv::Mat outImage;
  cv::drawKeypoints(im, keypoints, outImage );
  cv::imshow( "win-keys", outImage );
  cv::waitKey(0);


  return 0;
}
