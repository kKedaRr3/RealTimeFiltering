cmake_minimum_required (VERSION 3.18)

# Dodajemy CUDA od razu, żeby projekt był gotowy na Twoje filtry
project ("RealTimeFilteringMain" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# --- TU WPISZ SWOJE ŚCIEŻKI ---
# Przykłady:
# set(Qt6_DIR "C:/Qt/6.5.0/msvc2022_64/lib/cmake/Qt6")
# set(OpenCV_DIR "C:/opencv/build")
set(Qt6_DIR "A:/Qt/6.11.0/msvc2022_64/lib/cmake/Qt6")
set(OpenCV_DIR "A:/OpenCV/opencv/build/x64/vc16/lib")

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

add_subdirectory ("RealTimeFiltering")