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
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace std;


// Give a set of knn=2 raw matches, eliminates matches based on Lowe's Ratio test.
// Also returns the point features set.
// Cite: Lowe, David G. "Distinctive image features from scale-invariant keypoints." International journal of computer vision 60.2 (2004): 91-110.
void lowe_ratio_test( const vector<cv::KeyPoint>& keypoints1, const vector<cv::KeyPoint>& keypoints2 ,
                      const std::vector< std::vector< cv::DMatch > >& matches_raw,
                      vector<cv::Point2f>& pts_1, vector<cv::Point2f>& pts_2, float threshold=0.85 )
{
  pts_1.clear();
  pts_2.clear();
  assert( matches_raw.size() > 0 );
  assert( matches_raw[0].size() >= 2 );

  for( int j=0 ; j<matches_raw.size() ; j++ )
  {
    if( matches_raw[j][0].distance < threshold * matches_raw[j][1].distance ) //good match
    {
      // Get points
      int t = matches_raw[j][0].trainIdx;
      int q = matches_raw[j][0].queryIdx;
      pts_1.push_back( keypoints1[q].pt );
      pts_2.push_back( keypoints2[t].pt );
    }
  }

}


// Given 2 images with their matches points. ie. pts_1[i] <---> pts_2[i].
// This function returns the plotted with/without lines numbers
void drawMatches( const cv::Mat& im1, const vector<cv::Point2f>& pts_1,
                  const cv::Mat& im2, const vector<cv::Point2f>& pts_2,
                  cv::Mat& outImage,
                  const string msg = string( ""  ),
                  vector<uchar> status = vector<uchar>(),
                  bool enable_lines=true, bool enable_points=true, bool enable_test=true)
{
  assert( pts_1.size() == pts_2.size() );
  assert( im1.rows == im2.rows );
  assert( im2.cols == im2.cols );
  assert( im1.channels() == im2.channels() );

  cv::Mat row_im;
  cv::hconcat(im1, im2, row_im);

  if( row_im.channels() == 3 )
    outImage = row_im.clone();
  else
    cvtColor(row_im, outImage, CV_GRAY2RGB);


    // loop  over points
  for( int i=0 ; i<pts_1.size() ; i++ )
  {
    cv::Point2f p1 = pts_1[i];
    cv::Point2f p2 = pts_2[i];

    if( enable_points ) {
      cv::circle( outImage, p1, 4, cv::Scalar(0,0,255), -1 );
      cv::circle( outImage, p2+cv::Point2f(im1.cols,0), 4, cv::Scalar(0,0,255), -1 );
    }

    if( enable_test ) {
      cv::putText( outImage, to_string(i).c_str(), p1, cv::FONT_HERSHEY_COMPLEX_SMALL, .5, cv::Scalar(0,0,255) );
      cv::putText( outImage, to_string(i).c_str(), p2+cv::Point2f(im1.cols,0), cv::FONT_HERSHEY_COMPLEX_SMALL, .5, cv::Scalar(0,0,255) );
    }

    if( enable_lines ) {

      if( status.size() > 0 ) //if status is present
      {
        if( status[i] > 0 )
        {
          cv::line( outImage,  p1, p2+cv::Point2f(im1.cols,0), cv::Scalar(0,255,0) );
        }
        else
        {
          cv::line( outImage,  p1, p2+cv::Point2f(im1.cols,0), cv::Scalar(0,0,255) );;
        }
      }
      else // no status. Then make all lines blue
      {
        cv::line( outImage,  p1, p2+cv::Point2f(im1.cols,0), cv::Scalar(255,0,0) );
      }


    }


  }


  if( msg.length() > 0 ) {
    cv::putText( outImage, msg, cv::Point(5,50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,255) );
  }

}


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


  //
  // Matcher - FLAN (Approx NN)
  if(descriptors1.type()!=CV_32F)
  {
    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);
  }
  cv::FlannBasedMatcher matcher;
  std::vector< std::vector< cv::DMatch > > matches_raw;
  matcher.knnMatch( descriptors1, descriptors2, matches_raw, 2 );
  cout << "# Matches : " << matches_raw.size() << endl; //N
  cout << "# Matches[0] : " << matches_raw[0].size() << endl; //2


    //
    // Lowe's Ratio test
    vector<cv::Point2f> pts_1, pts_2;
    lowe_ratio_test( keypoints1, keypoints2, matches_raw, pts_1, pts_2 );
    cout << "# Retained (after ratio test): "<< pts_1.size() << endl; // == pts_2.size()

    cv::Mat im_out1;
    drawMatches( im1, pts_1, im2, pts_2, im_out1 );
    cv::imshow( "Matching after Ratio Test", im_out1 );


    // f-test. You might need to undistort point sets. But if precision is not needed, probably skip it.
    // Warning: This might be a bit off, since you need to give findFundamentalMat() undistorted keypoints. But anyways I dont care.
    vector<uchar> status;
    cv::findFundamentalMat(pts_1, pts_2, cv::FM_RANSAC, 5.0, 0.99, status);
    drawMatches( im1, pts_1, im2, pts_2, im_out1, "", status  );
    cv::imshow( "Matching after F-Test", im_out1 );



  cv::waitKey(0);
  return 0;
}
