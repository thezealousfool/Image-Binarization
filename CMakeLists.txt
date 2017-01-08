cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0012 NEW)
PROJECT(project)

FIND_PACKAGE( OpenCV REQUIRED )
MESSAGE("OpenCV version : ${OpenCV_VERSION}")
MESSAGE("OpenCV library directories: ${OpenCV_LIB_DIR}")
MESSAGE("OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV link directories: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

ADD_EXECUTABLE( kmeans kmeans.cpp )
TARGET_LINK_LIBRARIES( kmeans ${OpenCV_LIBS} )

ADD_EXECUTABLE( kmeans_hue_value kmeans_hue_value.cpp )
TARGET_LINK_LIBRARIES( kmeans_hue_value ${OpenCV_LIBS} )

ADD_EXECUTABLE( new_kmeans new_kmeans.cpp )
TARGET_LINK_LIBRARIES( new_kmeans ${OpenCV_LIBS} )

ADD_EXECUTABLE( kmeans_test kmeans_test.cpp )
TARGET_LINK_LIBRARIES( kmeans_test ${OpenCV_LIBS} )
