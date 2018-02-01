/*
    Reads Images and DAISY. Use case sample for daisy detector. This is avaialble in opencv3+

    Documentation http://docs.opencv.org/trunk/d9/d37/classcv_1_1xfeatures2d_1_1DAISY.html

    E. Tola, V. Lepetit, and P. Fua. DAISY: An Efficient Dense Descriptor Applied to Wide Baseline Stereo. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(5):815–830, May 2010

    Author : Manohar Kuse <mpkuse@connect.ust.hk>
    Released in Public Domain
*/

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
  //
  // Load Images
  cv::Mat im = cv::imread( "../image/church1.jpg");
  cout << "image dim : "<< im.rows << ", " << im.cols << ", " << im.channels() << endl;
  cv::imshow( "win", im );
  cv::waitKey(0);


  //
  // DAISY
  cv::Ptr<DAISY> daisy_detector = cv::xfeatures2d::DAISY::create();
  // cv::Ptr<SURF> detectr = SURF::create( 400 );
  cv::Mat whole_image_descriptor;
  daisy_detector->compute( im, whole_image_descriptor );

  cout << whole_image_descriptor.rows << ", " << whole_image_descriptor.cols << endl;
  // cout << daiso << endl;

  return 0;
}
