cmake_minimum_required(VERSION 2.8.3)
project(vfc)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

find_package(OpenCV 3 REQUIRED )
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_executable( keypoints keypoints.cpp )
add_executable( matcher matcher.cpp )

add_executable( robust_matcher_vfc vfc.cpp robust_matcher_vfc.cpp )
add_executable( robust_matcher robust_matcher.cpp )
add_executable( daisy daisy_keypoints.cpp )



target_link_libraries( keypoints  ${OpenCV_LIBRARIES} )
target_link_libraries( matcher  ${OpenCV_LIBRARIES} )
target_link_libraries( robust_matcher_vfc  ${OpenCV_LIBRARIES} )
target_link_libraries( robust_matcher  ${OpenCV_LIBRARIES} )
target_link_libraries( daisy  ${OpenCV_LIBRARIES} )
