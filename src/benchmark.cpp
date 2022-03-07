#include "Vulkan_mandelbrot_generator.hpp"
#include "Openmp_mandelbrot_generator.hpp"

#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Two parameters required." << "\n";
    return 1;
  }

  try {
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);

    std::cout << "Image size: " << width << " * " << height << "\n";

    {
      auto raw_image = std::vector<unsigned char>{};
      auto start = std::chrono::high_resolution_clock::now();
      raw_image = Vulkan_mandelbrot_generator{width, height}.generate();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << "Vulkan: " << duration << "ms\n";
    }

    {
      auto raw_image = std::vector<unsigned char>{};
      auto start = std::chrono::high_resolution_clock::now();
      raw_image = Openmp_mandelbrot_generator{width, height}.generate();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << "OpenMP: " << duration << "ms\n";
    }

    return std::cout ? 0 : 1;
  } catch (const vk::SystemError& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "unknown exception\n";
    return 1;
  }
}