cmake_minimum_required(VERSION 2.8.3)
project(vfc)


find_package(OpenCV REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_executable( keypoints keypoints.cpp )
add_executable( matcher matcher.cpp )
add_executable( robust_matcher vfc.cpp robust_matcher.cpp )
add_executable( daisy daisy_keypoints.cpp )


target_link_libraries( keypoints  ${OpenCV_LIBRARIES} )
target_link_libraries( matcher  ${OpenCV_LIBRARIES} )
target_link_libraries( robust_matcher  ${OpenCV_LIBRARIES} )
target_link_libraries( daisy  ${OpenCV_LIBRARIES} )
