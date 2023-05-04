cmake_minimum_required(VERSION 3.18)

project(test_project LANGUAGES C CXX)

# Find installed Open3D, which exports Open3D::Open3D
# find_package(glfw3 3.3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("openmp found")
endif()


find_package(Open3D REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(include)
include_directories( ${OpenCV_INCLUDE_DIRS} )


file(GLOB SOURCES "src/*.cpp")
message(${SOURCES})
add_executable(main ${SOURCES})


# add_executable(RegistrationBase)
target_sources(main PRIVATE src/main.cpp)

target_link_libraries(main PRIVATE Open3D::Open3D ${OpenCV_LIBS} OpenGL::GL)


# cmake .. -DCMAKE_BUILD_TYPE=Release
