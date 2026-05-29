cmake_minimum_required(VERSION 3.20)

project(RealTimeFiltering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(Qt6_DIR "C:/Qt/6.11.1/msvc2022_64/lib/cmake/Qt6")
set(OpenCV_DIR "C:/Users/matiu/Downloads/opencv/build/x64/vc16/lib")

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

add_executable(RealTimeFiltering
    "main.cpp"
    "RealTimeFiltering.cpp"
    "RealTimeFiltering.h"
    "CudaFilters.cu"
    "CudaFilters.h"
)

target_link_libraries(RealTimeFiltering PRIVATE
    Qt6::Widgets
    ${OpenCV_LIBS}
)

# Żeby Visual Studio uruchamiało program z folderu projektu
set_target_properties(RealTimeFiltering PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# OpenCV - kopiowanie DLL-ki
if(WIN32)
    set(OPENCV_DLL_NAME "opencv_world4110d.dll")

    get_filename_component(OPENCV_BIN_DIR "${OpenCV_DIR}/../bin" REALPATH)

    add_custom_command(TARGET RealTimeFiltering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_BIN_DIR}/${OPENCV_DLL_NAME}"
        "$<TARGET_FILE_DIR:RealTimeFiltering>"
        COMMENT "Kopiowanie OpenCV DLL do folderu exe..."
    )
endif()