cmake_minimum_required(VERSION 3.0.2)
project(cem_vis_planner)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  bebop_gazebo
  rospy
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cem_vis_planner
#  CATKIN_DEPENDS bebop_gazebo rospy
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

