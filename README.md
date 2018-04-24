# Robust Keypoint Matching with ORB Features

Basic demo of the papers

    [1] Jiayi Ma, Ji Zhao, Jinwen Tian, Alan Yuille, and Zhuowen Tu.<br/>
        Robust Point Matching via Vector Field Consensus,<br/>
        IEEE Transactions on Image Processing, 23(4), pp. 1706-1721, 2014<br/>
        
    [2] Jiayi Ma, Ji Zhao, Jinwen Tian, Xiang Bai, and Zhuowen Tu.<br/>
        Regularized Vector Field Learning with Sparse Approximation for Mismatch Removal,<br/>
        Pattern Recognition, 46(12), pp. 3519-3532, 2013<br/>


The original code supplied by this paper uses the SURF detectors which a lot of people have issues compiling due to the copyright issues with the SURF implementation. I adopted their original code to make it working with ORB features. 

This also contains usage samples for simple keypoint matching (with Lowe's ratio test and Fundamental-test for outlier rejection). Usage samples are in both C++ and Python. This can be used as a boilerplate code for many computer vision applications. Feature detection and matching is usually very common in a typical computer vision pipeline. 

This related [github-gist](https://gist.github.com/mpkuse/c96010112ec07269d944e199d029303a) might also be useful. 


# How to Compile
Only dependency is OpenCV
```
mkdir build
cmake ..
make
./robust_matcher
```
Python codes in `py` directory and can be executed directly. 

# Result
![Result Image VFC](image/result.png "Result Image VFC")

