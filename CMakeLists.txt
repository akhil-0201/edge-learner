cmake_minimum_required(VERSION 3.18)
project(edge_learner VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")

# ─── Dependencies ────────────────────────────────────────────────────────────
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc videoio imgcodecs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
pkg_check_modules(CURL REQUIRED libcurl)

# RKNN Lite2 (Rockchip RK3588)
# Set RKNN_ROOT to your rknn-toolkit2 runtime path, e.g.:
# cmake -DRKNN_ROOT=/opt/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api ...
set(RKNN_ROOT "${CMAKE_SOURCE_DIR}/third_party/rknn" CACHE PATH "Path to RKNN Lite2 runtime")
set(RKNN_INCLUDE_DIR "${RKNN_ROOT}/include")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(RKNN_LIB "${RKNN_ROOT}/lib/aarch64/librknnrt.so")
else()
  set(RKNN_LIB "${RKNN_ROOT}/lib/x86_64/librknnrt.so")
endif()

# nlohmann/json (header-only, vendored or system)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)
  FetchContent_MakeAvailable(json)
endif()

# ─── Sources ─────────────────────────────────────────────────────────────────
set(SOURCES
  src/main.cpp
  src/config/Settings.cpp
  src/inference/RKNNEngine.cpp
  src/inference/FramePipeline.cpp
  src/collector/BufferManager.cpp
  src/collector/ConfidenceMonitor.cpp
  src/trainer/TrainLauncher.cpp
  src/uploader/ModelUploader.cpp
)

add_executable(edge_learner ${SOURCES})

target_include_directories(edge_learner PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${RKNN_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${CURL_INCLUDE_DIRS}
)

target_link_libraries(edge_learner PRIVATE
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
  ${CURL_LIBRARIES}
  ${RKNN_LIB}
  nlohmann_json::nlohmann_json
  pthread
  stdc++fs
)

target_compile_options(edge_learner PRIVATE
  ${YAML_CPP_CFLAGS_OTHER}
  ${CURL_CFLAGS_OTHER}
)

# RPATH so it finds librknnrt.so at runtime
set_target_properties(edge_learner PROPERTIES
  INSTALL_RPATH "${RKNN_ROOT}/lib/aarch64"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# ─── Install ─────────────────────────────────────────────────────────────────
install(TARGETS edge_learner RUNTIME DESTINATION /opt/edge-learner/bin)
install(DIRECTORY config/ DESTINATION /opt/edge-learner/config)
install(FILES systemd/edge-learner-cpp.service
  DESTINATION /etc/systemd/system)

# ─── Tests ───────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests EXCLUDE_FROM_ALL)
