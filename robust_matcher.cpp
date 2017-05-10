/*
  a) Load 2 Images
  b) Detect ORB Keypoints for both images
  c) Compute ORB Descriptors
  d) FLANN Matcher
  e) Vector Field Consensus (VFC) for filtering false matches
  
  Note: 
    Depends on vfc.h and vfc.cpp
   
  Reference
    [1] Jiayi Ma, Ji Zhao, Jinwen Tian, Alan Yuille, and Zhuowen Tu.
        Robust Point Matching via Vector Field Consensus,
        IEEE Transactions on Image Processing, 23(4), pp. 1706-1721, 2014
    [2] Jiayi Ma, Ji Zhao, Jinwen Tian, Xiang Bai, and Zhuowen Tu.
        Regularized Vector Field Learning with Sparse Approximation for Mismatch Removal,
        Pattern Recognition, 46(12), pp. 3519-3532, 2013
        
    Author : Manohar Kuse <mpkuse@connect.ust.hk>
    Released in Public Domain
*/        
        
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "vfc.h"

using namespace std;

int main()
{
  //
  // Load Images
  cv::Mat im1 = cv::imread( "../image/church1.jpg");
  cv::Mat im2 = cv::imread( "../image/church2.jpg");

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


  // Matcher - FLAN (Approx NN)
  if(descriptors1.type()!=CV_32F)
  {
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);
  }
  cv::FlannBasedMatcher matcher;
  std::vector< cv::DMatch > matches;
  matcher.match( descriptors1, descriptors2, matches );

  //
  // Draw Matches
  cv::Mat outImg;
  cv::drawMatches(im1, keypoints1, im2, keypoints2, matches, outImg );
  cv::imshow( "Raw Matches", outImg );

  //
  // Filter Matches with Vector Field consensus (VFC)
  // a - preprocess data format
	vector<Point2f> X;
	vector<Point2f> Y;
	X.clear();
	Y.clear();
	for (unsigned int i = 0; i < matches.size(); i++) {
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		X.push_back(keypoints1[idx1].pt);
		Y.push_back(keypoints2[idx2].pt);
	}


  // b - main - vfc
  double t = (double)getTickCount();
	VFC myvfc;
	myvfc.setData(X, Y);
	myvfc.optimize();
	vector<int> matchIdx = myvfc.obtainCorrectMatch();
	t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
	cout << "Times (ms): " << t << endl;

  // c - post process
  std::vector< DMatch > correctMatches;
  std::vector<KeyPoint> correctKeypoints1, correctKeypoints2;
  correctMatches.clear();
  for (unsigned int i = 0; i < matchIdx.size(); i++) {
    int idx = matchIdx[i];
    correctMatches.push_back(matches[idx]);
    correctKeypoints1.push_back(keypoints1[idx]);
    correctKeypoints2.push_back(keypoints2[idx]);
  }

  //
  // Draw Corrected Matches
  Mat img_correctMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, correctMatches, img_correctMatches);
  imshow("Detected Correct Matches", img_correctMatches);


  cv::waitKey(0);
  return 0;
}

        