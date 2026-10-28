cmake_minimum_required(VERSION 3.16)

project(yunet_realtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

add_executable(yunet_realtime
    src/face_extractor/face_extractor.cpp
    src/face_label/FaceLabel.cpp
    src/inference/inference.cpp
    src/main.cpp
)

target_include_directories(yunet_realtime PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(yunet_realtime PRIVATE ${OpenCV_LIBS})

target_compile_options(yunet_realtime PRIVATE -Wall -Wextra)