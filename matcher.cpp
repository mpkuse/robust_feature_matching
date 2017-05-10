/*
  Reads 2 images, detects keypoints, compute ORB descriptors at these keypoints, 
  matches the descriptors by nearest neighbour (brute force)
  
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
  cv::Mat im1 = cv::imread( "../image/church1.jpg");
  cv::Mat im2 = cv::imread( "../image/church2.jpg");
  cv::imshow( "im1", im1 );
  cv::imshow( "im2", im2 );
  cv::waitKey(0);


  //
  // Feature Detector
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  fdetector->detectAndCompute(im1, cv::Mat(), keypoints1, descriptors1);
  fdetector->detectAndCompute(im2, cv::Mat(), keypoints2, descriptors2);
  cout << "# of keypoints : "<< keypoints1.size() << endl;
  cout << "# of keypoints : "<< keypoints2.size() << endl;
  cout << "descriptors shape : "<< descriptors1.rows << "x" << descriptors1.cols << endl;
  cout << "descriptors shape : "<< descriptors2.rows << "x" << descriptors2.cols << endl;


  //
  // Draw Keypoints
  cv::Mat outImage1, outImage2;
  cv::drawKeypoints(im1, keypoints1, outImage1 );
  cv::drawKeypoints(im2, keypoints2, outImage2 );
  cv::imshow( "win-keys1", outImage1 );
  cv::imshow( "win-keys2", outImage2 );
  cv::waitKey(0);

  //
  // Matcher - Brute Force
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector< cv::DMatch > matches;
  matcher.match(descriptors1, descriptors2, matches);

  /* - Consider using an FLANN based matches for speed
  // Matcher - FLANN (Approx NN)
  if(descriptors1.type()!=CV_32F)
  {
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);
  }
  cv::FlannBasedMatcher matcher;
  std::vector< cv::DMatch > matches;
  matcher.match( descriptors1, descriptors2, matches );
  */

  //
  // Draw Matches
  cv::Mat outImg;
  cv::drawMatches(im1, keypoints1, im2, keypoints2, matches, outImg );
  cv::imshow( "matches", outImg );
  cv::waitKey(0);

  return 0;
}
