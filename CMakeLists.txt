cmake_minimum_required(VERSION 3.16)

project(ZoneGuardAI
    VERSION 0.1.0
    DESCRIPTION "Human fall detection with monitoring zones - Qt desktop app"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt code generators (moc/uic/rcc).
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Qt: prefer Qt6, fall back to Qt5 -----------------------------------------
# The codebase is written to compile against either major version.
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
message(STATUS "Using Qt ${QT_VERSION_MAJOR} (${Qt${QT_VERSION_MAJOR}_VERSION})")

# --- OpenCV -------------------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "Using OpenCV ${OpenCV_VERSION}")

# --- Threads ------------------------------------------------------------------
find_package(Threads REQUIRED)

# --- ONNX Runtime -------------------------------------------------------------
# Use a system/SDK install if ONNXRUNTIME_ROOT is provided, otherwise download
# the official prebuilt release. Override the version with -DORT_VERSION=...
set(ORT_VERSION "1.17.1" CACHE STRING "ONNX Runtime version to download")

if(DEFINED ONNXRUNTIME_ROOT)
    set(ORT_ROOT "${ONNXRUNTIME_ROOT}")
    message(STATUS "Using ONNX Runtime from ONNXRUNTIME_ROOT=${ORT_ROOT}")
else()
    include(FetchContent)
    FetchContent_Declare(onnxruntime
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(ORT_ROOT "${onnxruntime_SOURCE_DIR}")
    message(STATUS "Using downloaded ONNX Runtime ${ORT_VERSION} at ${ORT_ROOT}")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# --- Sources ------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/core/detector.cpp
    src/core/pose_estimator.cpp
    src/core/kalman_filter.cpp
    src/core/tracker.cpp
    src/core/pipeline.cpp
    src/vision/frame_source.cpp
    src/ui/video_widget.cpp
    src/ui/main_window.cpp
)

set(HEADERS
    src/core/types.h
    src/core/detector.h
    src/core/pose_estimator.h
    src/core/kalman_filter.h
    src/core/tracker.h
    src/core/pipeline.h
    src/vision/frame_source.h
    src/ui/video_widget.h
    src/ui/main_window.h
)

add_executable(zoneguard ${SOURCES} ${HEADERS})

target_include_directories(zoneguard PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# Default location of the exported ONNX models (overridable at runtime via the
# ZONEGUARD_MODELS_DIR environment variable).
target_compile_definitions(zoneguard PRIVATE
    ZONEGUARD_MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models/onnx"
)

target_link_libraries(zoneguard PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    onnxruntime
    Threads::Threads
)

# Let the executable find libonnxruntime.so at runtime.
set_target_properties(zoneguard PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib"
)
