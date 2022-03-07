cmake_minimum_required(VERSION 3.21)
project(VulkanMandelbrot)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(MandelbrotGPU src/mandelbrot_gpu.cpp src/lodepng.cpp)
target_link_libraries(MandelbrotGPU Vulkan::Vulkan)

configure_file(shaders/comp.spv shaders/comp.spv COPYONLY)

add_executable(MandelbrotCPU src/mandelbrot_cpu.cpp src/lodepng.cpp)
target_link_libraries(MandelbrotCPU Vulkan::Vulkan glm::glm OpenMP::OpenMP_CXX)