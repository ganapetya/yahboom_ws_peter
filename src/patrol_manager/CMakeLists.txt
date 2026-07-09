cmake_minimum_required(VERSION 3.8)
project(patrol_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

include_directories(include)

add_library(waypoint_patrol_strategy src/waypoint_patrol_strategy.cpp)
ament_target_dependencies(waypoint_patrol_strategy rclcpp)

add_executable(patrol_manager_node src/patrol_manager_node.cpp)
target_link_libraries(patrol_manager_node waypoint_patrol_strategy ${OpenCV_LIBS})
ament_target_dependencies(patrol_manager_node
  rclcpp rclcpp_action nav2_msgs geometry_msgs sensor_msgs std_msgs
)

install(TARGETS
  waypoint_patrol_strategy
  patrol_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
