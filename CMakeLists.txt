cmake_minimum_required(VERSION 3.23)

include(CMakePackageConfigHelpers)

project(basalt-headers VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest) # note: this adds a BUILD_TESTING which defaults to ON

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
      message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
      set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
      set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    else()
      message(STATUS "Dind't find ccache")
    endif()
  else()
    message(STATUS "Compiler launcher already set. Not configuring ccache.")
    message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
    message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
  endif()

  IF( NOT CMAKE_BUILD_TYPE )
    SET( CMAKE_BUILD_TYPE Release)
  ENDIF()

  IF(NOT CXX_MARCH)
    SET(CXX_MARCH native)
  ENDIF()

  IF(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    set(BASALT_HEADERS_MARCH_FLAGS "-march=${CXX_MARCH}")
  ELSE()
    message(STATUS "Running on Apple ${CMAKE_SYSTEM_PROCESSOR}. Disabled -march flag.")
  ENDIF()


  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)


  # clang-specific compile flags
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(BASALT_HEADERS_CXX_FLAGS "${BASALT_HEADERS_CXX_FLAGS} -Wno-exceptions -fcolor-diagnostics -frelaxed-template-template-args -Wno-error=deprecated-declarations")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      # These are disabled to avoid lot's of warnings in Eigen code with clang 10
      set(BASALT_HEADERS_CXX_FLAGS "${BASALT_HEADERS_CXX_FLAGS} -Wno-misleading-indentation")
    endif()
  else()
    set(BASALT_HEADERS_CXX_FLAGS "${BASALT_HEADERS_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=array-bounds")
  endif()


  set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # cmake default: "-g"
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                      # cmake default: "-O3 -DNDEBUG"
  SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra ${BASALT_HEADERS_MARCH_FLAGS} ${CMAKE_CXX_FLAGS} ${BASALT_HEADERS_CXX_FLAGS}")
  SET(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline")
  SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
  SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
  set(CMAKE_CXX_FLAGS_SANITIZERDEBUG "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined")
  set(CMAKE_CXX_FLAGS_SANITIZERRELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-omit-frame-pointer -fsanitize=address,undefined")

endif()

find_package(Eigen3 3.4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(cereal REQUIRED)

add_library(basalt-headers INTERFACE)
add_library (basalt::basalt-headers ALIAS basalt-headers)
target_link_libraries(basalt-headers INTERFACE Eigen3::Eigen Sophus::Sophus cereal::cereal)

# Associate target with include directory
target_include_directories(basalt-headers INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_sources(basalt-headers
  PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        include/basalt/calibration/calib_bias.hpp
        include/basalt/calibration/calibration.hpp
        include/basalt/camera/bal_camera.hpp
        include/basalt/camera/camera_static_assert.hpp
        include/basalt/camera/double_sphere_camera.hpp
        include/basalt/camera/extended_camera.hpp
        include/basalt/camera/fov_camera.hpp
        include/basalt/camera/generic_camera.hpp
        include/basalt/camera/kannala_brandt_camera4.hpp
        include/basalt/camera/pinhole_camera.hpp
        include/basalt/camera/pinhole_radtan8_camera.hpp
        include/basalt/camera/stereographic_param.hpp
        include/basalt/camera/unified_camera.hpp
        include/basalt/image/image.h
        include/basalt/image/image_pyr.h
        include/basalt/imu/imu_types.h
        include/basalt/imu/preintegration.h
        include/basalt/serialization/eigen_io.h
        include/basalt/serialization/headers_serialization.h
        include/basalt/spline/ceres_local_param.hpp
        include/basalt/spline/ceres_spline_helper.h
        include/basalt/spline/rd_spline.h
        include/basalt/spline/se3_spline.h
        include/basalt/spline/so3_spline.h
        include/basalt/spline/spline_common.h
        include/basalt/utils/assert.h
        include/basalt/utils/eigen_utils.hpp
        include/basalt/utils/hash.h
        include/basalt/utils/sophus_utils.hpp
)

# NOTE: The target "basalt-headers" has to be installed by any project using this project
#       in order to install the headers.

if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR BASALT_HEADERS_BUILD_TESTING) AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
