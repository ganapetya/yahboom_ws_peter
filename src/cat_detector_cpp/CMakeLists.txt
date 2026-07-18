cmake_minimum_required(VERSION 3.8)
project(cat_detector_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc imgcodecs)

add_executable(detector_node src/detector_node.cpp)
target_link_libraries(detector_node ${OpenCV_LIBS})
ament_target_dependencies(detector_node
  rclcpp sensor_msgs std_msgs vision_msgs
)

install(TARGETS
  detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
