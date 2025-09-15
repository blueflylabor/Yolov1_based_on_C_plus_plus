cmake_minimum_required(VERSION 3.10)
project(YOLOv1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加编译器选项，确保能找到标准库头文件
if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /Library/Developer/CommandLineTools/usr/include/c++/v1")
endif()

# 设置OpenCV路径，如果需要的话
# find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加源文件
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bbox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo.cpp
)

# 创建库
add_library(yolov1_lib ${SOURCES})

# 创建可执行文件
add_executable(yolov1_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/demo.cpp)
target_link_libraries(yolov1_demo yolov1_lib)

# 创建训练示例可执行文件
add_executable(yolov1_train ${CMAKE_CURRENT_SOURCE_DIR}/examples/train_example.cpp)
target_link_libraries(yolov1_train yolov1_lib)

# 创建真实数据集训练可执行文件
add_executable(yolov1_train_real ${CMAKE_CURRENT_SOURCE_DIR}/examples/train_real_data.cpp)
target_link_libraries(yolov1_train_real yolov1_lib)

# 创建CIFAR-10数据集转换可执行文件
add_executable(yolov1_cifar10_to_yolo ${CMAKE_CURRENT_SOURCE_DIR}/examples/cifar10_to_yolo.cpp)
target_link_libraries(yolov1_cifar10_to_yolo yolov1_lib)

# 创建CIFAR-10评估可执行文件
add_executable(yolov1_cifar10_evaluate ${CMAKE_CURRENT_SOURCE_DIR}/examples/cifar10_evaluate.cpp)
target_link_libraries(yolov1_cifar10_evaluate yolov1_lib)

# 创建MNIST数据集转换可执行文件
add_executable(yolov1_mnist_to_yolo ${CMAKE_CURRENT_SOURCE_DIR}/examples/mnist_to_yolo.cpp)
target_link_libraries(yolov1_mnist_to_yolo yolov1_lib)

# 如果使用OpenCV，取消下面的注释
# target_link_libraries(yolov1_demo yolov1_lib ${OpenCV_LIBS})
# target_link_libraries(yolov1_train_real yolov1_lib ${OpenCV_LIBS})

# 安装规则
install(TARGETS yolov1_lib yolov1_demo yolov1_train yolov1_train_real yolov1_cifar10_to_yolo yolov1_cifar10_evaluate yolov1_mnist_to_yolo
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)