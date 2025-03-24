cmake_minimum_required(VERSION 3.31)

project(infini_train VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add gflags
add_subdirectory(third_party/gflags)
include_directories(${gflags_SOURCE_DIR}/include)

# Add glog
add_subdirectory(third_party/glog)
include_directories(${glog_SOURCE_DIR}/src)

# Add infini_train
include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
add_library(infini_train STATIC ${SRC})
target_link_libraries(infini_train glog gflags)

# Examples
add_executable(mnist example/mnist/main.cc example/mnist/dataset.cc example/mnist/net.cc)
target_link_libraries(mnist glog gflags infini_train)
