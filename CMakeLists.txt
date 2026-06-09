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

# --- Sources ------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/vision/FrameSource.cpp
    src/ui/VideoWidget.cpp
    src/ui/MainWindow.cpp
)

set(HEADERS
    src/vision/FrameSource.h
    src/ui/VideoWidget.h
    src/ui/MainWindow.h
)

add_executable(zoneguard ${SOURCES} ${HEADERS})

target_include_directories(zoneguard PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(zoneguard PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    Threads::Threads
)
