cmake_minimum_required(VERSION 3.21)
project(VulkanMandelbrot)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(mandelbrot src/mandelbrot.cpp src/lodepng.cpp src/Vulkan_mandelbrot_generator.cpp
  src/Openmp_mandelbrot_generator.cpp)
target_link_libraries(mandelbrot Vulkan::Vulkan glm::glm OpenMP::OpenMP_CXX)

add_executable(benchmark src/benchmark.cpp src/Vulkan_mandelbrot_generator.cpp src/Openmp_mandelbrot_generator.cpp)
target_link_libraries(benchmark Vulkan::Vulkan glm::glm OpenMP::OpenMP_CXX)

configure_file(shaders/comp.spv shaders/comp.spv COPYONLY)