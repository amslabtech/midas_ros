cmake_minimum_required(VERSION 3.0.2)
project(midas_ros)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  rospy
)
